from domain_config.model_conf import *
import yaml


def training_config_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Config(**fields)


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_yaml(filename: str):
    loader = yaml.SafeLoader
    loader.add_constructor('!TrainingConfig', training_config_constructor)
    loader.add_constructor('!CNNModelConfig', training_config_constructor)

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data
