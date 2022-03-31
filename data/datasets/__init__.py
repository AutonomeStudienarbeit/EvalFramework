import os

import pandas as pd
import numpy as np

from data.datasets.available_datasets import available_datasets
from re import match as match_regex


class DatasetLoader:

    def __init__(self):
        from os import path as os_path, getcwd
        self.__location__ = os_path.realpath(os_path.join(getcwd(), os_path.dirname(__file__)))

        self.dataset_apis = {
            "kaggle": self.download_from_kaggle,
        }

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

    def unzip_dataset(self, file_name):
        from zipfile import ZipFile
        path = self.__location__ + f"/{file_name[:-4]}"
        z = ZipFile(self.__location__ + f"/{file_name}")
        z.extractall(self.__location__ + f"/{file_name[:-4]}")
        return path

    def load_dataset(self, dataset_name):
        if dataset_name not in available_datasets.keys():
            raise ValueError(f"The dataset of name {dataset_name} is unavailable")

        dir_content = self.get_directory_content()
        dataset_name_lower = dataset_name.lower()
        file_name = next((entry for entry in dir_content if f"{dataset_name_lower}" == entry), None)
        zip_file = next((entry for entry in dir_content if f"{dataset_name_lower}.zip" in entry), None)
        if zip_file is None:
            zip_file = self.install_dataset(dataset_name)
        if file_name is None:
            dataset_path = self.unzip_dataset(zip_file)
        else:
            dataset_path = f"{self.__location__}/{dataset_name_lower}"

        return Dataset(dataset_name, dataset_path)


class Dataset:

    def __init__(self, dataset_id, dataset_path):
        from os import path as os_path, getcwd
        self.path = dataset_path
        self.dataset_id = dataset_id
        self.dataset_properties = available_datasets.get(dataset_id)
        self.description = self.dataset_properties.get("description")
        self.url = self.dataset_properties.get("url")
        self.version = self.dataset_properties.get("version")
        self.dataset_id = dataset_id
        self.__location__ = os_path.realpath(os_path.join(getcwd(), os_path.dirname(__file__)))
        self.folders = [entry for entry in os.listdir(self.path) if match_regex("^(?!.*[.]).*", entry)]
        self.train_ground_truth = self.dataset_properties.get("gt_train_path")
        self.test_ground_truth = self.dataset_properties.get("gt_test_path")

        self._train_subset = None
        self._test_subset = None
        self._validation_subset = None

        if not self.dataset_properties.get("annotated_test"):
            self._train_subset, self._validation_subset, self._test_subset = self._create_splits()

    def _create_splits(self):

        # load available image filenames
        train_dir = f"{self.path}/{self.dataset_properties.get('train_folder_path')}"
        filenames = [f"{train_dir}/{image}" for image in os.listdir(train_dir) if
                     match_regex(".*([.](ppm|png|jpg))", image)]
        filenames = pd.DataFrame(filenames)

        # create train, val & test splits
        return np.split(filenames.sample(frac=1, random_state=42),
                        [int(.6 * len(filenames)),
                         int(.8 * len(filenames))])  # train: 80%, val: 20%, test: 20%

    def load_validation_subset(self):
        if self._validation_subset is not None:
            return self._validation_subset

        val_dir = f"{self.path}/{self.dataset_properties.get('val_folder_path')}"
        if self.dataset_properties.get("val_folder_path") is not None:
            subset_list = []
            for entry in os.listdir(val_dir):
                if match_regex(".*([.](ppm|png|jpg))", entry):
                    subset_list.append(f"{val_dir}/{entry}")
                elif entry != "yolo" and not match_regex(".*([.](csv|txt))", entry):
                    subset_list.extend([f"{val_dir}/{entry}/{image}" for image in os.listdir(f"{val_dir}/{entry}")])
                self._validation_subset = pd.DataFrame(
                    subset_list
                )
            return self._validation_subset
        else:
            raise FileNotFoundError("Could not find Test / test folder in dataset")

    def load_train_subset(self):
        if self._train_subset is not None:
            return self._train_subset

        train_dir = f"{self.path}/{self.dataset_properties.get('train_folder_path')}"
        if self.dataset_properties.get("train_folder_path") is not None:
            subset_list = []
            for entry in os.listdir(train_dir):
                if match_regex(".*([.](ppm|png|jpg))", entry):
                    subset_list.append(f"{train_dir}/{entry}")
                elif entry != "yolo" and not match_regex(".*([.](csv|txt))", entry):
                    subset_list.extend([f"{train_dir}/{entry}/{image}" for image in os.listdir(f"{train_dir}/{entry}")])
            self._train_subset = pd.DataFrame(
                subset_list
            )
            return self._train_subset
        else:
            raise FileNotFoundError("Could not find Train / train folder in dataset")

    def load_test_subset(self):
        if self._test_subset is not None:
            return self._test_subset

        test_dir = f"{self.path}/{self.dataset_properties.get('test_folder_path')}"
        if self.dataset_properties.get("test_folder_path") is not None:
            subset_list = []
            for entry in os.listdir(test_dir):
                if match_regex(".*([.](ppm|png|jpg))", entry):
                    subset_list.append(f"{test_dir}/{entry}")
                elif entry != "yolo" and not match_regex(".*([.](csv|txt))", entry):
                    subset_list.extend([f"{test_dir}/{entry}/{image}" for image in os.listdir(f"{test_dir}/{entry}")])
                self._test_subset = pd.DataFrame(
                    subset_list
                )
            return self._test_subset
        else:
            raise FileNotFoundError("Could not find Test / test folder in dataset")
