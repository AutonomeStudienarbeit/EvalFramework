import os
from unittest import TestCase
from data.models.fasterRCNN import FasterRCNN
from data.datasets import DatasetLoader


class FasterRCNNTests(TestCase):

    # TODO: Test it
    def test_dataset_prep_gtsdb(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        fasterRCNN = FasterRCNN()
        fasterRCNN.prepare_dataset(gtsdb_dataset, "train")

        fasterRCNN.torch_dataset.__getitem__(0)

        for file in os.listdir(f"{gtsdb_dataset.path}/fasterRCNN/train/images"):
            self.assertEqual(file[-4:], ".png", "File Endings do not match .png")
