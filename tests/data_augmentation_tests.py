from unittest import TestCase
from os import listdir, getcwd, path

from data.datasets import DatasetLoader
from data.datasets.data_augmentation import DataAugmentation


class DataAugmentationTests(TestCase):
    datasets_folder = path.realpath(getcwd() + "/../data/datasets/")

    def test_blur(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        data_augmentation.blur_set(radius=4, frac=0.5)
