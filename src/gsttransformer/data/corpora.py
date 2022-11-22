import os

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import re
from .utils import DataSetSplit

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from typing import List, Union, Optional, Dict, Pattern


class IEMOCAP(Dataset):
    CORPUS_ID: str = 'IEMOCAP_full_release'
    VALID_LINE_REGEX: Pattern[str] = re.compile(
        r'^((Ses(\d+)[MF]_(impro|script)\d+(_\d)?)_([MF])\d+) \[\d{3}\.\d{4}-\d{3}\.\d{4}\]: (.+)$'
    )
    ENTRIES = ('session_idx', 'transcript_id', 'line_id', 'session_type', 'speaker_gender', 'utterance')
    ENTRIES_IDXS = (2, 1, 0, 3, 5, 6)

    def __init__(
            self,
            corpus_dir_path: str,
            data_set_split: str,
            *args,
            validation_size: Optional[Union[int, float]] = None,
            test_size: Optional[Union[int, float]] = None,
            random_seed: Optional[int] = None,
            concurrent_backend: str = 'threading',
            n_jobs: int = -1,
            verbosity_level: int = 2,
            **kwargs
    ):
        super(IEMOCAP, self).__init__()
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Load data
        # Get file list
        file_list: List[str] = [
            os.path.join(corpus_dir_path, ses_dir, 'dialog', 'transcriptions', transcripts_file)
            for ses_dir in (
                ses_dir for ses_dir in os.listdir(corpus_dir_path)
                if ses_dir.startswith('Session') and os.path.isdir(os.path.join(corpus_dir_path, ses_dir))
            )
            for transcripts_file in os.listdir(os.path.join(corpus_dir_path, ses_dir, 'dialog', 'transcriptions'))
            if transcripts_file.endswith('.txt')
        ]
        # Get indices list
        idxs = range(len(file_list))
        # Do train/validation/test split on the indices
        train_idxs, test_idxs = train_test_split(idxs, test_size=test_size, random_state=random_seed)
        train_idxs, validation_idxs = train_test_split(train_idxs, test_size=validation_size, random_state=random_seed)
        # Load the desired split
        if self.data_set_split == DataSetSplit.TRAIN:
            idxs = train_idxs
        elif self.data_set_split == DataSetSplit.VALIDATION:
            idxs = validation_idxs
        elif self.data_set_split == DataSetSplit.TEST:
            idxs = test_idxs
        else:
            raise ValueError(f"Unsupported data split: {self.data_set_split.value}")
        # Parallelisation options
        self.parallel_backend: str = concurrent_backend
        self.n_jobs: int = n_jobs
        self.verbosity_level: int = verbosity_level
        # Load selected split and add actions placeholders
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            self.data: List[List[Dict]] = Parallel(verbose=self.verbosity_level)(
                delayed(self._load_txt_file)(os.path.join(file_list[idx])) for idx in idxs
            )

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int) -> List[Dict]:
        return self.data[index]

    def _load_txt_file(self, path: str) -> List[Dict]:
        def create_sample_dict(line: str) -> Dict:
            tmp = {
                key: self.VALID_LINE_REGEX.findall(line.strip())[0][value_idx]
                for key, value_idx in zip(self.ENTRIES, self.ENTRIES_IDXS)
            }
            tmp['audio_file_path'] = os.path.join(
                self.CORPUS_ID,
                f"Session{int(tmp['session_idx'])}",
                'sentences',
                'wav',
                tmp['transcript_id'],
                f"{tmp['line_id']}.wav"
            )

            return tmp

        with open(path) as f:
            lines = f.read().strip().split('\n')

        return [
            create_sample_dict(line)
            for line in lines if self.VALID_LINE_REGEX.match(line.strip())
        ]
