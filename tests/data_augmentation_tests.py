import os
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

    def test_gaussian_noise(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        data_augmentation.perturb_set_gaussian_noise(frac=0.5)

    def test_salt_pepper(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        data_augmentation.perturb_set_salt_pepper(frac_of_set=0.5, frac_of_images=0.5)

    def test_image_brightness(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        data_augmentation.perturb_set_image_brightness(frac=0.5, brightness=125)

    def test_block(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        data_augmentation.block_set(frac=0.5, color=(0, 255, 0))

    def test_add_sticker(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
        data_augmentation.add_stickers_to_set(1)

    def test_load_augmentation(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="test")
        path = data_augmentation.blur_set(radius=4, frac=0.5)
        gtsdb_dataset.set_augmentated_set(path)
        train_set = set(gtsdb_dataset.load_test_subset()[0])
        self.assertSetEqual(set([f"{path}/{image}" for image in os.listdir(path)]), train_set, "Subsets not Equal")
