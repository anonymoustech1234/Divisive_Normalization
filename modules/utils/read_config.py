import os
import json

def read_config(path: str) -> dict:
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("File doesn't exist:", path)
    
def read_config_in_dir(root_dir: str, file_name: str) -> dict:
    path = os.path.join(root_dir, file_name)
    return read_config(path)