from data.datasets import DatasetLoader
from unittest import TestCase
from os import listdir, getcwd, path


class DatasetTests(TestCase):
    datasets_folder = path.realpath(getcwd() + "/../data/datasets/")

    def test_gtsrb(self):
    #     if "gtsrb.zip" in listdir(self.datasets_folder):
    #         remove(self.datasets_folder + "/gtsrb.zip")

        dataset_loader = DatasetLoader()
        gtsrb_dataset = dataset_loader.load_dataset("GTSRB")
        self.assertTrue("gtsrb.zip" in listdir(self.datasets_folder),
                        "gtsrb.zip not found!")
        train_subset = gtsrb_dataset.load_train_subset()
        print(train_subset.iloc[0][0])
        self.assertGreater(len(train_subset), 0, "Length of train subset is not greater than 0")
        test_subset = gtsrb_dataset.load_test_subset()
        self.assertGreater(len(test_subset), 0, "Length of test subset is not greater than 0")

    def test_gtsdb(self):
        # if "gtsdb.zip" in listdir(self.datasets_folder):
        #     remove(self.datasets_folder + "/gtsdb.zip")

        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        self.assertTrue("gtsdb.zip" in listdir(self.datasets_folder),
                        "gtsdb.zip not found!")
        train, val, test = gtsdb_dataset.load_train_subset(), gtsdb_dataset.load_validation_subset(), gtsdb_dataset.load_test_subset()
        print(train)
        self.assertEqual(360, len(train), "Length of train subset does not match")
        self.assertEqual(120, len(val), "Length of val subset does not match")
        self.assertEqual(120, len(test), "Length of test subset does not match")
