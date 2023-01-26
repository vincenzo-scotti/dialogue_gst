from enum import Enum

IGNORE_INDEX = -1


class DataSetSplit(Enum):
    DEVELOPMENT: str = 'dev'
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'


class EncodingMode(Enum):
    RESPONSE_AND_CONTEXT: str = 'ctx_resp'
    RESPONSE_ONLY: str = 'resp'
    RESPONSE_FROM_CONTEXT: str = 'resp_from_ctx'
