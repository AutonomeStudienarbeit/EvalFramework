from pathlib import Path


def create_nested_folders(*paths):
    for path in paths:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)


def get_key_by_value_of_list(dictionary, value):
    keys = dictionary.keys()
    for key in keys:
        if value in dictionary.get(key):
            return key
