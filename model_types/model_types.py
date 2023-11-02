from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D, SpatialDropout1D, GlobalMaxPooling1D
from domain_config.model_conf import *
from abc import ABC


class AbstractModel(ABC):
    def __init__(self):
        self.n_unique_words = 0

    def build_model(self):
        pass


class CnnModel(AbstractModel):
    def __int__(
            self,
            model_config: CNNModelConfig,
            training_config: TrainingConfig
    ):
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = 0
        self.max_review_length = 0
        self.drop_embed = 0
        self.n_conv = 0
        self.k_conv = 0
        self.n_dense = 0
        self.dropout = 0

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.n_unique_words,
                            self.n_dim,
                            input_length=self.max_review_length))
        model.add(SpatialDropout1D(self.drop_embed))

        model.add(Conv1D(self.n_conv, self.k_conv, activation='relu'))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(self.n_dense, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(1, activation="sigmoid"))

        return model


def build_cnn_model(model_config: CNNModelConfig, training_config: TrainingConfig):
    # 3b) Define CONVOLUTIONAL network architecture
    # In addition to what the dense network is doing, i.e. associating
    # words to sentiment, convolutional networks are able to also
    # consider the order/sequence of certain words to predict sentiment
    # e.g. not-good would be understood by the model to be similar as
    # not-great as good and great are likely to share a similar location
    # in the vector space

    assert type(model_config) == CNNModelConfig, (f"incorrect model config type received, "
                                                  f"expected CNNModelConfig but received "
                                                  f"{type(model_config)}")

    model = Sequential()
    model.add(Embedding(training_config.n_unique_words,
                        training_config.n_dim,
                        input_length=training_config.max_review_length))
    model.add(SpatialDropout1D(training_config.drop_embed))

    model.add(Conv1D(model_config.n_conv, model_config.k_conv, activation='relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(model_config.n_dense, activation="relu"))
    model.add(Dropout(model_config.dropout))

    model.add(Dense(1, activation="sigmoid"))

    return model
