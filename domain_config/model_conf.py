from enum import Enum


class TrainingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    epoch: int = 4
    batch_size: int = 128
    n_dim: int = 64

    n_unique_words: int = 5000
    n_words_to_skip: int = 50
    max_review_length: int = 400
    pad_type: str = "pre"
    trunc_type: str = "pre"
    drop_embed: float = 0.2

    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    metrics: list = ["accuracy"]


class ModelTypes(Enum):
    CNN = "CNN"
    dense = "dense"


class CnnModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    n_conv: int = 256
    k_conv: int = 3
    n_dense: int = 256
    dropout: float = 0.2


class DenseModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    n_dense: int = 64
    dropout: float = 0.5


class LstmModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    n_lstm: int = 256
    dropout: float = 0.2


class RnnModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    n_rnn: int = 256
    dropout: float = 0.2


class StackedBiLstmModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    n_lstm_1: int = 256
    n_lstm_2: int = 256
    dropout: float = 0.2
