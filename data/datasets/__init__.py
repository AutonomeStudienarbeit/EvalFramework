class DatasetLoader:

    def __init__(self):
        from json import load
        self.available_datasets = load(open('datasets.json'))

    @staticmethod
    def download_from_kaggle(dataset_properties):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_properties.get("download_url"), quiet=False)
        out_file_name = f"{dataset_properties.get('download_url').split('/')[1]}.zip"
        return out_file_name

    def install_dataset(self, dataset_name):
        dataset_properties = self.available_datasets.get(dataset_name)
        if dataset_properties.get("is_kaggle"):
            return self.download_from_kaggle(dataset_properties)

    @staticmethod
    def get_directory_content():
        from os import listdir
        return listdir("./")

    def load_dataset(self, dataset_name):
        if dataset_name not in self.available_datasets.keys():
            raise ValueError(f"The dataset of name {dataset_name} is unavailable")

        dir_content = self.get_directory_content()
        dataset_name_lower = dataset_name.lower()
        file_name = next((entry for entry in dir_content if dataset_name_lower in entry), None)
        if file_name is None:
            file_name = self.install_dataset(dataset_name)
        return Dataset(dataset_name, file_name, self.available_datasets.get(dataset_name))


class Dataset:

    def __init__(self, dataset_id, zip_file_name, dataset_properties):
        from zipfile import ZipFile
        from re import match as match_regex
        self.description = dataset_properties.get("description")
        self.url = dataset_properties.get("url")
        self.version = dataset_properties.get("version")
        self.dataset_id = dataset_id
        self.ZipFile = ZipFile(zip_file_name)
        self.zip_content = self.ZipFile.namelist()
        toplevel = {entry.split("/")[0] for entry in self.zip_content}
        self.folders = {entry for entry in toplevel if match_regex("^(?!.*[.]).*", entry)}
        self.description_csvs = {entry for entry in toplevel if match_regex(".*([.]csv)", entry)}


dataset_loader = DatasetLoader()
dataset_loader.load_dataset("GTSRB")
