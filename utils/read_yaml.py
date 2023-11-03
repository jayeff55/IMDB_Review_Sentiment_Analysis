from domain_config.model_conf import *
import yaml


def training_config_constructor(loader, node):
    fields = loader.construct_mapping(node)

    if node.tag == "!TrainingConfig":
        return TrainingConfig(**fields)
    elif node.tag == "!CnnModelConfig":
        return CnnModelConfig(**fields)
    elif node.tag == "!DenseModelConfig":
        return DenseModelConfig(**fields)
    elif node.tag == "!LstmModelConfig":
        return LstmModelConfig(**fields)
    elif node.tag == "!MultiCnnModelConfig":
        return MultiCnnModelConfig(**fields)
    elif node.tag == "!RnnModelConfig":
        return RnnModelConfig(**fields)
    elif node.tag == "!StackedBiLstmModelConfig":
        return StackedBiLstmModelConfig(**fields)
    else:
        raise TypeError("Config type not recognised")


def load_yaml(filename: str):
    loader = yaml.SafeLoader
    loader.add_constructor('!TrainingConfig', training_config_constructor)
    loader.add_constructor('!CnnModelConfig', training_config_constructor)
    loader.add_constructor('!DenseModelConfig', training_config_constructor)
    loader.add_constructor('!LstmModelConfig', training_config_constructor)
    loader.add_constructor('!MultiCnnModelConfig', training_config_constructor)
    loader.add_constructor('!RnnModelConfig', training_config_constructor)
    loader.add_constructor('!StackedBiLstmModelConfig', training_config_constructor)

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data
