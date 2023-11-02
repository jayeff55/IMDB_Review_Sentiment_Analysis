from enum import Enum


class TrainingConfig:
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


class CNNModelConfig:
    n_conv: int = 256
    k_conv: int = 3
    n_dense: int = 256
    dropout: float = 0.2


class DenseModelConfig:
    a = 1