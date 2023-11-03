from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Embedding
from keras.models import Model
from keras.layers import Conv1D, SpatialDropout1D, GlobalMaxPooling1D, LSTM, SimpleRNN, Input, concatenate
from keras.layers.wrappers import Bidirectional
from utils.model_conf import *
from abc import ABC


class AbstractModel(ABC):
    def __init__(self):
        self.n_unique_words = 0
        self.n_dim = 0
        self.max_review_length = 0
        self.drop_embed = 0
        self.model = None

    def build_model(self):
        pass


class CnnModel(AbstractModel):
    def __init__(
            self,
            model_config: CnnModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_conv = model_config.n_conv
        self.k_conv = model_config.k_conv
        self.n_dense = model_config.n_dense
        self.dropout = model_config.dropout

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
        self.model = model


class DenseModel(AbstractModel):
    def __init__(
            self,
            model_config: DenseModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_dense = model_config.n_dense
        self.dropout = model_config.dropout

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.n_unique_words,
                            self.n_dim,
                            input_length=self.max_review_length))

        model.add(Flatten())

        model.add(Dense(self.n_dense, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(self.n_dense, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(1, activation="sigmoid"))
        self.model = model


class LstmModel(AbstractModel):
    def __init__(
            self,
            model_config: LstmModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_lstm = model_config.n_lstm
        self.dropout = model_config.dropout

    def build_model(self):
        model = Sequential()

        model.add(Embedding(self.n_unique_words,
                            self.n_dim,
                            input_length=self.max_review_length))
        model.add(SpatialDropout1D(self.drop_embed))

        model.add(LSTM(self.n_lstm, dropout=self.dropout))

        model.add(Dense(1, activation="sigmoid"))
        self.model = model


class NonSeqMultiConvModel(AbstractModel):
    def __init__(
            self,
            model_config: MultiCnnModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_dense = model_config.n_dense
        self.dropout = model_config.dropout
        self.n_conv_1 = model_config.n_conv_1
        self.n_conv_2 = model_config.n_conv_2
        self.n_conv_3 = model_config.n_conv_3
        self.k_conv_1 = model_config.k_conv_1
        self.k_conv_2 = model_config.k_conv_2
        self.k_conv_3 = model_config.k_conv_3

    def build_model(self):
        # Define input layer
        input_layer = Input(shape=self.max_review_length,
                            dtype="int16",
                            name="input")

        # Define embedding layer
        embedding_layer = Embedding(self.n_unique_words,
                                    self.n_dim,
                                    name="embedding")(input_layer)
        drop_embed_layer = SpatialDropout1D(self.drop_embed,
                                            name="drop_embed")(embedding_layer)

        # Define 3 parallel Conv2D streams
        conv_1 = Conv1D(self.n_conv_1,
                        self.k_conv_1,
                        activation="relu",
                        name="conv_1")(drop_embed_layer)
        maxp_1 = GlobalMaxPooling1D(name="maxp_1")(conv_1)

        conv_2 = Conv1D(self.n_conv_2,
                        self.k_conv_2,
                        activation="relu",
                        name="conv_2")(drop_embed_layer)
        maxp_2 = GlobalMaxPooling1D(name="maxp_2")(conv_2)

        conv_3 = Conv1D(self.n_conv_3,
                        self.k_conv_3,
                        activation="relu",
                        name="conv_3")(drop_embed_layer)
        maxp_3 = GlobalMaxPooling1D(name="maxp_3")(conv_3)

        # Concatenate outputs/activations from the 3 parallel streams above
        concat = concatenate([maxp_1, maxp_2, maxp_3])

        # Define 2 dense layers
        dense_1 = Dense(self.n_dense,
                        activation="relu",
                        name="dense_1")(concat)
        drop_dense_1 = Dropout(self.dropout,
                               name="dropout_1")(dense_1)

        dense_2 = Dense(int(self.n_dense / 4),
                        activation="relu",
                        name="dense_2")(drop_dense_1)
        drop_dense_2 = Dropout(self.dropout,
                               name="dropout_2")(dense_2)

        # Define sigmoid layer
        predictions = Dense(1,
                            activation="sigmoid",
                            name="output")(drop_dense_2)

        # Finally, assemble the two parts together!
        model = Model(input_layer, predictions)
        self.model = model


class RnnModel(AbstractModel):
    def __init__(
            self,
            model_config: RnnModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_rnn = model_config.n_rnn
        self.dropout = model_config.dropout

    def build_model(self):
        model = Sequential()

        model.add(Embedding(self.n_unique_words,
                            self.n_dim,
                            input_length=self.max_review_length))
        model.add(SpatialDropout1D(self.drop_embed))

        model.add(SimpleRNN(self.n_rnn, dropout=self.dropout))
        # No dense layer added after a recurrent layer because it provides little
        # performance advantage

        model.add(Dense(1, activation="sigmoid"))
        self.model = model


class StackedBiLstmModel(AbstractModel):
    def __init__(
            self,
            model_config: StackedBiLstmModelConfig,
            training_config: TrainingConfig
    ):
        super().__init__()
        self.n_unique_words = training_config.n_unique_words
        self.n_dim = training_config.n_dim
        self.max_review_length = training_config.max_review_length
        self.drop_embed = training_config.drop_embed
        self.n_lstm_1 = model_config.n_lstm_1
        self.n_lstm_2 = model_config.n_lstm_1
        self.dropout = model_config.dropout

    def build_model(self):
        model = Sequential()

        model.add(Embedding(self.n_unique_words,
                            self.n_dim,
                            input_length=self.max_review_length))
        model.add(SpatialDropout1D(self.drop_embed))

        model.add(Bidirectional(LSTM(self.n_lstm_1,
                                     dropout=self.dropout,
                                     return_sequences=True)))
        model.add(Bidirectional(LSTM(self.n_lstm_2,
                                     dropout=self.dropout)))

        model.add(Dense(1, activation="sigmoid"))
        self.model = model
