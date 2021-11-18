def download_from_kaggle(dataset):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(available_datasets.get(dataset).get("download_url"), quiet=False)


def install_dataset(dataset_name):
    dataset_properties = available_datasets.get(dataset_name)
    if dataset_properties.get("is_kaggle"):
        download_from_kaggle(dataset_properties.get("download_url"))


def get_directory_content():
    from os import listdir
    return listdir("./")


def check_already_available(dataset_name):
    dataset_name = dataset_name.lower()
    dir_content = get_directory_content()
    for file in dir_content:
        if dataset_name in file:
            return True
    return False


def load_dataset(dataset_name):
    if dataset_name not in available_datasets.keys():
        raise ValueError("The dataset of name {} is unavailable".format(dataset_name))
    if not check_already_available(dataset_name):
        install_dataset(dataset_name)

    from zipfile import ZipFile
    dir_content = get_directory_content()
    dataset_name_lower = dataset_name.lower()
    for entry in dir_content:
        if dataset_name_lower in entry:
            dataset = ZipFile.open(entry)
    dataset.namelist()

from json import load
available_datasets = load(open('datasets.json'))

load_dataset("GTSRB")
