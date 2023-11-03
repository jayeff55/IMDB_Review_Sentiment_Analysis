import sys

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from domain_config.model_conf import *
from utils.read_yaml import *
import yaml
import numpy as np


def load_process_imdb_data(training_config: TrainingConfig):
    # 2. Loading and pre-processing data
    (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=training_config.n_unique_words,
                                                        skip_top=training_config.n_words_to_skip)

    # convert review from index integers to natural language
    word_index = keras.datasets.imdb.get_word_index()

    # resetting index value (value + 3 to accommodate additional entries below)
    word_index = {k : (v+3) for k, v in word_index.items()}

    # to the dictionary of word_index, add 3 new entries
    # which are customary for representing padding, starting and unknown tokens
    word_index["PAD"] = 0
    word_index["START"] = 1
    word_index["UNK"] = 2

    # invert the dictionary
    #index_word = {v: k for k, v in word_index.items()}

    # standardise length of reviews
    x_train = pad_sequences(x_train,
                            maxlen=training_config.max_review_length,
                            padding=training_config.pad_type,
                            truncating=training_config.trunc_type,
                            value=0)

    x_val = pad_sequences(x_val,
                          maxlen=training_config.max_review_length,
                          padding=training_config.pad_type,
                          truncating=training_config.trunc_type,
                          value=0)

    return x_train, y_train, x_val, y_val


def compile_model(training_config: TrainingConfig, model):
    model.compile(optimizer=training_config.optimizer,
                  loss=training_config.loss,
                  metrics=training_config.metrics)


def fit_model(
        model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        training_config: TrainingConfig,
        model_checkpoint: keras.callbacks.callbacks.ModelCheckpoint
):
    model.fit(x_train,
              y_train,
              batch_size=training_config.batch_size,
              epochs=training_config.epoch,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[model_checkpoint])
