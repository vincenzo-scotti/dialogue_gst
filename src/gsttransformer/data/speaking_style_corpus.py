import os

import bz2
import pickle

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils import DataSetSplit, EncodingMode
from .corpora import IEMOCAP

from transformers import GPT2Tokenizer, GPT2Model

from mellotron_api import load_tts, get_gst_scores, get_gst_embeddings

from typing import List, Tuple, Dict, Optional, Union

CORPORA: Dict = {
    IEMOCAP.CORPUS_ID: IEMOCAP
}


# TODO move here context cutting, it shouldn't be part of the corpora loaders
# TODO add class for fine tuning
class GSTTCorpus(Dataset):
    def __init__(
            self,
            corpora_dir_path: str,
            gpt2: Union[str, GPT2Model],
            tokenizer: Union[str, GPT2Tokenizer],
            mellotron: Union[str, Tuple],
            data_set_split: str,
            cache_dir_path: str,
            encoding_mode: str,
            *args,
            corpus_prefix: str = 'dgst_corpus',
            corpus_list: Optional[List[str]] = None,
            reload_cache: bool = False,
            max_context_length: Optional[int] = None,
            max_response_length: Optional[int] = None,
            response_prefix_token: Optional[str] = None,
            response_suffix_token: Optional[str] = None,
            gst_embeds: bool = True,
            gst_scores: bool = True,
            in_mem: int = 1,
            device: Optional[torch.device] = None,
            mixed_precision: bool = True,
            concurrent_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 2,
            **kwargs
    ):
        super(GSTTCorpus, self).__init__()
        #
        self.in_mem: int = in_mem
        self.device: torch.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.mixed_precision: bool = mixed_precision
        # Text
        # Model
        self.gpt2: GPT2Model = GPT2Model.from_pretrained(gpt2) if isinstance(gpt2, str) else gpt2
        # Tokeniser to prepare inputs
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Max lengths
        self.max_context_length: Optional[int] = max_context_length
        self.max_response_length: Optional[int] = max_response_length
        # Special tokens
        self.response_prefix_token: Optional[str] = response_prefix_token if response_prefix_token is not None else ''
        self.response_suffix_token: Optional[str] = response_suffix_token if response_suffix_token is not None else self.tokenizer.eos_token
        # Speech
        # Model
        self.mellotron, self.mellotron_stft, self.mellotron_hparams = load_tts(mellotron) if isinstance(mellotron, str) else mellotron
        # Data
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Encoding mode
        self.encoding_mode: EncodingMode = EncodingMode(encoding_mode)
        # Path to corpora data frame
        self.corpus_cache_file_path: str = os.path.join(cache_dir_path, f'{corpus_prefix}_{data_set_split}.pbz2')
        # Data
        self.data: List[Dict]
        # Flags
        self.gst_embeds: bool = gst_embeds
        self.gst_scores: bool = gst_scores

        # Generate cache if needed
        if not os.path.exists(self.corpus_cache_file_path) or reload_cache:
            # Save parallelisation options
            self.parallel_backend: str = concurrent_backend
            self.n_jobs: int = n_jobs
            self.verbosity_level: int = verbosity_level
            # Create cache dir if not exists
            if not os.path.exists(cache_dir_path):
                os.mkdir(cache_dir_path)
            #
            self.corpora_dir_path: str = corpora_dir_path
            # Get corpora list ad list of all available corpora if not provided
            if corpus_list is None:
                self.corpus_list: List[str] = [
                    dir_name for dir_name in os.listdir(corpora_dir_path)
                    if os.path.isdir(os.path.join(corpora_dir_path, dir_name))
                ]
            # Else simply save the provided list
            else:
                self.corpus_list: List[str] = corpus_list
            # Load all corpora and generate cache
            self._generate_data_cache(*args, **kwargs)
        # Else simply load the cache
        else:
            self._load_data_cache()

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return self.data[index]

    def _get_dialogue_contexts(self, dialogue_turns: List[str]) -> List[str]:
        # Gather all available context strings
        context_strings: List[str] = [self.tokenizer.bos_token] + [turn + self.tokenizer.eos_token for turn in dialogue_turns[:-1]]
        # If a limit on the context is given cut it
        if self.max_context_length is not None:
            # Accumulator for contexts
            dialogue_contexts: List[List[int]] = list()
            # Tokenise context strings
            tokenized_context_strings: List[List[int]] = self.tokenizer(context_strings).input_ids
            # Lenght of context strings
            tokenized_context_string_lengths: List[int] = [
                len(tokenized_context_string) for tokenized_context_string in tokenized_context_strings
            ]
            # For each response, select the context of maximum allowed length
            for e_idx in range(1, len(dialogue_turns) + 1):
                tmp_tokenized_context: List[List[int]] = tokenized_context_strings[:e_idx]
                tmp_tokenized_context_lengths: List[int] = tokenized_context_string_lengths[:e_idx]
                tmp_all_tokenized_context_length: int = sum(tmp_tokenized_context_lengths)
                while tmp_all_tokenized_context_length > self.max_context_length and len(tmp_tokenized_context) > 1:
                    tmp_all_tokenized_context_length -= tmp_tokenized_context_lengths.pop(0)
                    tmp_tokenized_context.pop(0)
                dialogue_contexts.append(sum(tmp_tokenized_context, [])[-self.max_context_length:])

            # Convert tokenised contexts into a list of strings
            dialogue_context_strings: List[str] = self.tokenizer.batch_decode(dialogue_contexts)
        # Else simply concatenate all context strings up to a point
        else:
            dialogue_context_strings: List[str] = [sum(context_strings[:i+1])for i in range(len(context_strings))]

        return dialogue_context_strings

    def _generate_data_cache(self, *args, **kwargs):
        # Helper function to prepare actual data
        def _preprocess_dialogue(original_dialogue):
            # Dialogue contexts
            dialogue_contexts = self._get_dialogue_contexts(list(turn['utterance'] for turn in original_dialogue))
            # Add context to dialogue
            for turn, context in zip(original_dialogue, dialogue_contexts):
                turn['context'] = context

        # Create corpora instances
        corpora = [
            CORPORA[corpus_id](
                os.path.join(self.corpora_dir_path, corpus_id),
                self.data_set_split.value,
                *args,
                parallel_backend=self.parallel_backend,
                n_jobs=self.n_jobs,
                verbosity_level=self.verbosity_level,
                **kwargs
            )
            for corpus_id in self.corpus_list
        ]
        # Generate and cut contexts and responses
        self.data = [corpus[idx] for corpus in corpora for idx in range(len(corpus))]
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            Parallel(verbose=self.verbosity_level)(delayed(_preprocess_dialogue)(dialogue) for dialogue in self.data)
        # and flatten the data
        self.data = [sample for dialogue in self.data for sample in dialogue]
        # Compute GST vectors
        for sample in self.data:
            sample['gst_embeddings'] = get_gst_embeddings(
                os.path.join(self.corpora_dir_path, sample['audio_file_path']),
                self.mellotron, self.mellotron_stft, self.mellotron_hparams
            )
        # Compute GST scores
        for sample in self.data:
            sample['gst_scores'] = get_gst_scores(
                os.path.join(self.corpora_dir_path, sample['audio_file_path']),
                self.mellotron, self.mellotron_stft, self.mellotron_hparams
            )
        # Save compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'w') as f:
            pickle.dump(self.data, f)
        # Generate and cut contexts and responses # NOTE this is not part of the cache
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            self._compute_contextual_embeddings()

    def _load_data_cache(self):
        # Load compressed pickle file
        with bz2.BZ2File(self.corpus_cache_file_path, 'r') as f:
            self.data = pickle.load(f)
        # Generate and cut contexts and responses # NOTE this is not part of the cache
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
            self._compute_contextual_embeddings()

    @torch.no_grad()
    def _compute_contextual_embeddings(self):
        # Iterate over mini batches
        for s_idx in range(len(self.data)):
            e_idx = min(len(self.data), s_idx + self.in_mem)
            # Gather current samples
            mini_batch = [self.data[data_idx] for data_idx in range(s_idx, e_idx)]
            # Prepare inputs depending on context embedding approach
            input_encodings = self.tokenizer([
                ('' if self.encoding_mode == EncodingMode.RESPONSE_ONLY else sample['context']) +
                self.response_prefix_token + sample['utterance'] + self.response_suffix_token
                for sample in mini_batch
            ], return_tensors='pt', padding=True).to(self.device)
            # Prepare valid positions maks
            valid_mask = input_encodings.attention_mask.bool()
            if self.encoding_mode == EncodingMode.RESPONSE_FROM_CONTEXT:
                for b_idx, sample in enumerate(mini_batch):
                    valid_mask[b_idx, :len(self.tokenizer(sample['context']).input_ids)] = False
            # Compute embeddings
            hidden = self.gpt2(**input_encodings).last_hidden_state
            # Retrieve resulting embeddings
            for batch_idx, data_idx in enumerate(range(s_idx, e_idx)):
                self.data[data_idx]['embeddings'] = hidden[batch_idx, valid_mask[batch_idx]].unsqueeze(0).cpu()

    def collate(self, mini_batch) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # Get max length for padding
        max_len = max(sample['embeddings'].size(1) for sample in mini_batch)
        # Create batch with current embeddings
        input_embeds = torch.vstack([
            F.pad(sample['embeddings'], (0, 0, 0, max_len - sample['embeddings'].size(1))) for sample in mini_batch
        ])
        # Get attention mask
        attention_mask = torch.vstack([F.pad(
            torch.ones((len(mini_batch), sample['embeddings'].size(1)), dtype=torch.long),
            (0, max_len - sample['embeddings'].size(1))
        ) for sample in mini_batch])
        # GST embeddings
        gst_embeddings = torch.tensor([sample['gst_embeddings'] for sample in mini_batch]) if self.gst_embeds else None
        # GST scores
        gst_scores = torch.tensor([sample['gst_scores'] for sample in mini_batch]) if self.gst_scores else None

        return input_embeds, attention_mask, gst_embeddings, gst_scores
