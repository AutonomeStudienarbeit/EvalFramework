from data.datasets.available_datasets import available_datasets


class DatasetLoader:

    def __init__(self):
        from os import path as os_path, getcwd
        self.__location__ = os_path.realpath(os_path.join(getcwd(), os_path.dirname(__file__)))

        self.dataset_apis = {
            "kaggle": self.download_from_kaggle,
            "cityscapesscripts": self.download_cityscapes
        }

    def download_cityscapes(self, dataset_properties):
        from os import system as run_in_terminal
        cs_package_id = dataset_properties.get("download_url")
        run_in_terminal("cd " + self.__location__ + "; " + "csDownload " + cs_package_id)
        return cs_package_id

    def download_from_kaggle(self, dataset_properties):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_properties.get("download_url"), quiet=False, path=self.__location__)
        out_file_name = f"{dataset_properties.get('download_url').split('/')[1]}.zip"
        return out_file_name

    def install_dataset(self, dataset_name):
        from os import rename
        dataset_properties = available_datasets.get(dataset_name)
        file_name = self.dataset_apis.get(dataset_properties.get("download_api"))(dataset_properties)
        dataset_name_lower = dataset_name.lower()
        rename(f"{self.__location__}/{file_name}", f"{self.__location__}/{dataset_name_lower}.zip")
        return f"{dataset_name_lower}.zip"

    def get_directory_content(self):
        from os import listdir
        return listdir(self.__location__)

    def load_dataset(self, dataset_name):
        if dataset_name not in available_datasets.keys():
            raise ValueError(f"The dataset of name {dataset_name} is unavailable")

        dir_content = self.get_directory_content()
        dataset_name_lower = dataset_name.lower()
        file_name = next((entry for entry in dir_content if dataset_name_lower in entry), None)
        if file_name is None:
            file_name = self.install_dataset(dataset_name)
            required_dataset = available_datasets.get(dataset_name).get("required_data")
            if required_dataset != "":
                self.install_dataset(required_dataset)
        return Dataset(dataset_name)


class Dataset:

    def __init__(self, dataset_id):
        from os import path as os_path, getcwd
        self.dataset_id = dataset_id
        self.dataset_properties = available_datasets.get(dataset_id)
        self.description = self.dataset_properties.get("description")
        self.url = self.dataset_properties.get("url")
        self.version = self.dataset_properties.get("version")
        self.dataset_id = dataset_id
        self.__location__ = os_path.realpath(os_path.join(getcwd(), os_path.dirname(__file__)))
        self.folders, self.csv_files, self.zip_content = self.load_zip(dataset_id)

        if self.dataset_properties.get("required_data") != "":
            folders_required_set, csv_files_required_set, zip_content_required_set = self.load_zip(self.dataset_properties.get("required_data"))
            self.folders, self.csv_files, self.zip_content = [*self.folders, *folders_required_set], [*self.csv_files, *csv_files_required_set], [*self.zip_content, *zip_content_required_set]

        self._train_subset = None
        self._test_subset = None

    def load_train_subset(self):
        if self._train_subset:
            return self._train_subset

        if "train" in self.folders:
            self._train_subset = self.get_subset_by_folder("train")
            return self._train_subset
        elif "Test" in self.folders:
            self._train_subset = self.get_subset_by_folder("Train")
            return self._train_subset
        else:
            raise FileNotFoundError("Could not find Train / train folder in dataset")

    def load_test_subset(self):
        if self._test_subset:
            return self._test_subset

        if "test" in self.folders:
            self._test_subset = self.get_subset_by_folder("test")
            return self._test_subset
        elif "Test" in self.folders:
            self._test_subset = self.get_subset_by_folder("Test")
            return self._test_subset
        else:
            raise FileNotFoundError("Could not find Test / test folder in dataset")

    def get_subset_by_folder(self, folder):
        return {entry for entry in self.zip_content if entry.split("/")[0] == folder}

    def load_zip(self, dataset_id):
        from zipfile import ZipFile
        from re import match as match_regex
        from os import path
        zip_file_name = dataset_id.lower() + ".zip"
        zip_file = ZipFile(path.join(self.__location__, zip_file_name))
        zip_content = zip_file.namelist()
        toplevel = {entry.split("/")[0] for entry in zip_content}
        folders = [entry for entry in toplevel if match_regex("^(?!.*[.]).*", entry)]
        csv_files = [entry for entry in toplevel if match_regex(".*([.]csv)", entry)]
        return folders, csv_files, zip_content
