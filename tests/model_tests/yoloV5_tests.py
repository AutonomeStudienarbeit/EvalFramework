import os
from unittest import TestCase
from os import path, getcwd, listdir

from data.datasets import DatasetLoader
from data.models.yolov5 import YoloV5

class YoloV5Tests(TestCase):
    datasets_folder = path.realpath(getcwd() + "/../../data/datasets/")

    def test_gtsrb_dataset_prep(self):
        dataset_loader = DatasetLoader()
        gtsrb_dataset = dataset_loader.load_dataset("GTSRB")
        yolov5 = YoloV5()
        yolov5.prepare_dataset(gtsrb_dataset)
        self.assertEqual(
            39209,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Train/yolo/images/")),
            "Length of train images folder does not match"
        )
        self.assertEqual(
            39209,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Train/yolo/labels/")),
            "Length of train labels folder does not match"
        )
        self.assertEqual(
            12631,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Test/yolo/images/")),
            "Length of test images folder does not match"
        )
        self.assertEqual(
            12630,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Test/yolo/labels/")),
            "Length of test labels folder does not match"
        )

    def test_gtsdb_dataset_prep(self):
        dataset_loader = DatasetLoader()
        gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
        yoloV5 = YoloV5()
        yoloV5.prepare_dataset(gtsdb_dataset)
        train_image_folder_content = os.listdir(f"{self.datasets_folder}/gtsdb/yolo/train/images/")
        self.assertEqual(
            360,
            len(train_image_folder_content),
            "Length of train images folder does not match"
        )
        self.assertEqual(".jpg", train_image_folder_content[0][-4:], "incorrect image format")
        self.assertEqual(
            360,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/yolo/train/labels/")),
            "Length of train labels folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/yolo/test/images/")),
            "Length of test images folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/yolo/test/labels/")),
            "Length of test labels folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/yolo/val/images/")),
            "Length of val images folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/yolo/val/labels/")),
            "Length of val labels folder does not match"
        )