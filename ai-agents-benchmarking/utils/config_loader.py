import yaml

def load_config(config_file="config/settings.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
