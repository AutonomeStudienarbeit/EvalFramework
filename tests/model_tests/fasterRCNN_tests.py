import os
from unittest import TestCase
from data.models.fasterRCNN import FasterRCNN
from data.datasets import DatasetLoader
from data.models.fasterRCNN.torchDataset import TorchDataset


class FasterRCNNTests(TestCase):

    def test_dataset_prep_gtsdb(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        # fasterRCNN = FasterRCNN(gtsdb_dataset)
        # fasterRCNN.prepare_dataset(gtsdb_dataset, "train")

        torchDataset = TorchDataset(gtsdb_dataset, "train")

        # fasterRCNN.torch_dataset.__getitem__(0)

        targets = []
        for i in range(304):
            print("index ", i)
            image, target = torchDataset.__getitem__(i)
            targets.append(target)

        for file in os.listdir(f"{gtsdb_dataset.path}/fasterRCNN/train/images"):
            self.assertEqual(file[-4:], ".png", "File Endings do not match .png")
