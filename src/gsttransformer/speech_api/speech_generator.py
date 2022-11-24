import torch
from transformers import GPT2Tokenizer, GPT2Model
from mellotron_api import load_tts, load_vocoder, load_arpabet_dict, synthesise_speech

from gsttransformer.model import GSTTransformer

from typing import Tuple, List, Union, Literal, Optional


class ChatSpeechGenerator:
    def __init__(
            self,
            gstt: Union[str, GSTTransformer],
            tokenizer: Union[str, GPT2Tokenizer],
            gpt2: Union[str, GPT2Model],
            mellotron: Optional[Union[str, Tuple]] = None,
            tacotron2: Optional[Union[str, Tuple]] = None,
            vocoder: Optional[Union[str, Tuple]] = None,
            arpabet_dict: Optional[Union[str, object]] = None,
            prefix_token: str = '',
            suffix_token: str = '<|endoftext|>',
            encoding_mode: Optional[Literal['resp', 'resp_from_ctx', 'ctx_resp']] = 'resp_from_ctx',
            device: Optional[torch.device] = None,
            mixed_precision: bool = True,
            max_context_len: Optional[int] = None
    ):
        super(ChatSpeechGenerator, self).__init__()
        # Dialogue Language Model
        self.gpt2: GPT2Model = GPT2Model.from_pretrained(gpt2).eval() if isinstance(gpt2, str) else gpt2
        # Tokenizer
        tokenizer = tokenizer if tokenizer is not None else (gpt2 if isinstance(gpt2, str) else self.gpt2.config._name_or_path)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        # Conditioned speech synthesis gpt2
        if mellotron is not None:
            if isinstance(mellotron, str):
                self.mellotron, self.mellotron_stft, self.mellotron_hparams = load_tts(mellotron)
            else:
                self.mellotron, self.mellotron_stft, self.mellotron_hparams = mellotron
        else:
            self.mellotron = self.mellotron_stft = self.mellotron_hparams = None
        # Raw TTS gpt2
        if tacotron2 is not None:
            if isinstance(tacotron2, str):
                self.tacotron2, self.tacotron2_stft, self.tacotron2_hparams = load_tts(tacotron2, model='tacotron2')
            else:
                self.tacotron2, self.tacotron2_stft, self.tacotron2_hparams = tacotron2
        else:
            self.tacotron2 = self.tacotron2_stft = self.tacotron2_hparams = None
        # Vocoder and denoiser
        if vocoder is not None:
            self.waveglow, self.denoiser = load_vocoder(vocoder) if isinstance(vocoder, str) else vocoder
        else:
            self.waveglow = self.denoiser = None
        # Arpabet dictionary
        if arpabet_dict is not None:
            self.arpabet_dict = load_arpabet_dict(arpabet_dict) if isinstance(arpabet_dict, str) else arpabet_dict
        else:
            self.arpabet_dict = None

        # GST predictor
        if gstt is not None and mellotron is not None:  # NOTE no point in having predictor if there is no conditioned TTS
            if isinstance(gstt, str):
                self.gstt = GSTTransformer(
                    self.gpt2.config,
                    self.mellotron.gst.stl.attention.num_units,
                    (self.mellotron.gst.stl.attention.num_heads, self.mellotron.gst.stl.embed.size(0))
                )
                self.gstt.load_state_dict(torch.load(gstt))
                self.gstt.eval().to(device)
            else:
                self.gstt = gstt
        else:
            self.gstt = None

        # Special tokens
        self.prefix_token = prefix_token
        self.suffix_token = suffix_token
        # text processing function parameters
        self.max_context_len: Optional[int] = max_context_len
        # GST processing function parameters
        self.encoding_mode = encoding_mode
        # Other low level settings
        self.device: torch.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision: bool = mixed_precision and self.device.type == 'cuda'
        # Load internal neural network gpt2 for text analysis
        self.gpt2 = self.gpt2.to(device)

    def __call__(self, *args, **kwargs):
        return self.generate_speech_response(*args, **kwargs)

    def _get_input_hidden_states(self, response: str, dialogue: Optional[List[str]] = None) -> torch.tensor:
        # Input validation
        assert self.gpt2 is not None and self.tokenizer is not None
        # Prepare context (if needed)
        if self.encoding_mode == 'resp_from_ctx' or self.encoding_mode == 'ctx_resp':
            context = self.tokenizer.bos_token + (
                ''.join(utterance + self.tokenizer.eos_token for utterance in dialogue) if dialogue is not None else ''
            )
        else:
            context = ''
        # Encode the input
        input_encodings = self.tokenizer(
            context + self.prefix_token + response + self.suffix_token,
            return_tensors='pt'
        ).to(self.device)
        # Prepare valid mask to retrieve the desired embeddings
        valid_mask = input_encodings.attention_mask.bool().squeeze()
        if self.encoding_mode == 'resp_from_ctx':
            valid_mask[:len(self.tokenizer(context).input_ids)] = False
        # Compute transformer last hidden states
        last_hidden_states = self.gpt2(**input_encodings).last_hidden_state[:, valid_mask]

        return last_hidden_states

    def _predict_gst(
            self, response: str, gst_prediction: Literal['embed', 'score'], dialogue: Optional[List[str]] = None
    ) -> Tuple[Optional[List[float]], Optional[List[List[float]]]]:
        # Input validation
        assert self.gstt is not None
        # Prepare input hidden states
        hidden_states = self._get_input_hidden_states(response, dialogue=dialogue)
        # Compute GST
        outputs = self.gstt(hidden_states)
        # Extract computed GST
        gst_embeddings = outputs['gst_embeds'].squeeze().cpu().tolist() if gst_prediction == 'embed' else None
        gst_scores = torch.softmax(outputs['gst_scores'], dim=-1).squeeze().cpu().tolist() if gst_prediction == 'score' else None

        return gst_embeddings, gst_scores

    def _synthesise_audio(
            self,
            text: str,
            speaker_id: Optional[int],
            gst_embeddings: Optional[List[float]],
            gst_scores: Optional[List[List[float]]],
            output_file_path: str
    ):
        # Input validation
        assert not (gst_embeddings is not None or gst_scores is not None or speaker_id is not None) or self.mellotron is not None
        # Generate audio
        if gst_embeddings is not None or gst_scores is not None or speaker_id is not None:
            # If conditioning is required use Mellotron to synthesise
            synthesise_speech(
                text,
                self.mellotron,
                self.mellotron_hparams,
                self.mellotron_stft,
                arpabet_dict=self.arpabet_dict,
                waveglow=self.waveglow,
                denoiser=self.denoiser,
                tacotron2=self.tacotron2,
                tacotron2_stft=self.tacotron2_stft,
                tacotron2_hparams=self.tacotron2_hparams,
                speaker_id=speaker_id,
                gst_head_style_scores=gst_scores,
                gst_style_embedding=gst_embeddings,
                out_path=output_file_path
            )
        else:
            # Else use directly Tacotron 2
            synthesise_speech(
                text,
                self.tacotron2,
                self.tacotron2_hparams,
                self.tacotron2_stft,
                arpabet_dict=self.arpabet_dict,
                waveglow=self.waveglow,
                denoiser=self.denoiser,
                out_path=output_file_path
            )

    # NOTE it is suggested to wrap the call il a temporary file context manager
    def generate_speech_response(
            self,
            response: str,
            output_file_path: str,
            dialogue: Optional[List[str]] = None,
            gst_prediction: Optional[Literal['embed', 'score']] = None,
            speaker_id: Optional[int] = None,
    ) -> str:
        # Predict GST embedding or scores from text (if required)
        if gst_prediction is not None:
            with torch.no_grad(), torch.autocast(self.device.type, enabled=self.mixed_precision):
                gst_embeddings, gst_scores = self._predict_gst(response, gst_prediction, dialogue=dialogue)
        else:
            gst_embeddings = gst_scores = None
        # Finally generate the audio clip
        self._synthesise_audio(response, speaker_id, gst_embeddings, gst_scores, output_file_path)
        # Return path of the synthesised audio clip
        return output_file_path

