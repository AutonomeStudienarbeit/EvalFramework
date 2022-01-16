from data.datasets import DatasetLoader
from unittest import TestCase
from os import listdir, getcwd, path, remove


class dataset_tests(TestCase):

    datasets_folder = path.realpath(getcwd() + "/../data/datasets/")

    def test_gtsrb(self):
        if "gtsrb.zip" in listdir(self.datasets_folder):
            remove(self.datasets_folder + "/gtsrb.zip")

        dataset_loader = DatasetLoader()
        gtsrb_dataset = dataset_loader.load_dataset("GTSRB")
        self.assertTrue("gtsrb.zip" in listdir(self.datasets_folder),
                        "gtsrb.zip not found!")
        self.assertSetEqual(gtsrb_dataset.folders, {'Test', 'test', 'train', 'Train', 'Meta', 'meta'},
                            "Folder Structure of gtsrb.zip not as expected!")
        self.assertSetEqual(gtsrb_dataset.csv_files, {'Train.csv', 'Meta.csv', 'Test.csv'},
                            "CSV Files of gtsrb.zip not as expected")
        train_subset = gtsrb_dataset.load_train_subset()
        self.assertGreater(len(train_subset), 0, "Length of train subset is not greater than 0")
        test_subset = gtsrb_dataset.load_test_subset()
        self.assertGreater(len(test_subset), 0, "Length of test subset is not greater than 0")

    def test_cityscapes_fine(self):
        if "gtFine_trainvaltest.zip" in listdir(self.datasets_folder):
            remove(self.datasets_folder + "/gtFine_trainvaltest.zip")
        dataset_loader = DatasetLoader()
        cs_fine_dataset = dataset_loader.load_dataset("Cityscapes-Fine")
        self.assertTrue("cityscapes-fine.zip" in listdir(self.datasets_folder),
                        "cityscapes-fine.zip not found!")
        print(cs_fine_dataset.folders)
