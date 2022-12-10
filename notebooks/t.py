import json
import os
def load_config_from_file(file_path):
    """Load the config from a file"""
    with open(file_path) as fp:
        json_data = json.load(fp)

    return json_data

if __name__=="__main__":
    path = "models"
    config = load_config_from_file(os.path.join(path, "config.json"))

    print(config)
