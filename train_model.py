import argparse
import sys
from pathlib import Path
from utils.training import *
from model_types.model_types import *
from keras.callbacks import ModelCheckpoint
import os


def main(training_config_filepath, model_config_filepath, output_directory):
    training_config = load_yaml(training_config_filepath)
    model_config = load_yaml(model_config_filepath)
    x_train, y_train, x_val, y_val = load_process_imdb_data(training_config)

    if type(model_config) == CNNModelConfig:
        model = build_cnn_model(model_config, training_config)
    else:
        model = Sequential()

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model_checkpoint = ModelCheckpoint(filepath=output_directory + "\\weights.{epoch:02d}.hdf5")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    model.fit(x_train,
              y_train,
              batch_size=batchSize,
              epochs=E,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[model_checkpoint])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_config_filepath", type=str, help="path to training config yaml")
    parser.add_argument("-m", "--model_config_filepath", type=str, help="path to model config yaml")
    parser.add_argument("-o", "--output_directory", type=str, help="output directory for trained model")
    args = parser.parse_args()

    if len(sys.argv) > 1:
        training_config_filepath = Path(args.training_config_filepath)
        model_config_filepath = Path(args.model_config_filepath)
        output_directory = Path(args.output_directory)
    else:
        training_config_filepath = Path("configs/training_configs.yaml")
        model_config_filepath = Path("configs/model_configs/CNN_configs.yaml")
        output_directory = Path("/trained_models")

    main(training_config_filepath, model_config_filepath, output_directory)
