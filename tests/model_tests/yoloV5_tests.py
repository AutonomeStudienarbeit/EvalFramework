import os
from unittest import TestCase
from os import path, getcwd, listdir
from data.models.yolov5 import YoloV5

class YoloV5Tests(TestCase):
    datasets_folder = path.realpath(getcwd() + "/../../data/datasets/")

    def test_gtsrb_dataset_prep(self):
        yolov5 = YoloV5()
        yolov5.prepare_dataset("gtsrb")
        self.assertEqual(
            39209,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Train/images/")),
            "Length of train images folder does not match"
        )
        self.assertEqual(
            39209,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Train/labels/")),
            "Length of train labels folder does not match"
        )
        self.assertEqual(
            12631,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Test/images/")),
            "Length of test images folder does not match"
        )
        self.assertEqual(
            12630,
            len(os.listdir(f"{self.datasets_folder}/gtsrb/Test/labels/")),
            "Length of test labels folder does not match"
        )

    def test_gtsdb_dataset_prep(self):
        yoloV5 = YoloV5()
        yoloV5.prepare_dataset('gtsdb')
        train_image_folder_content = os.listdir(f"{self.datasets_folder}/gtsdb/train/images/")
        self.assertEqual(
            360,
            len(train_image_folder_content),
            "Length of train images folder does not match"
        )
        self.assertEqual(".jpg", train_image_folder_content[0][-4:], "incorrect image format")
        self.assertEqual(
            360,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/train/labels/")),
            "Length of train labels folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/test/images/")),
            "Length of test images folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/test/labels/")),
            "Length of test labels folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/val/images/")),
            "Length of val images folder does not match"
        )
        self.assertEqual(
            120,
            len(os.listdir(f"{self.datasets_folder}/gtsdb/val/labels/")),
            "Length of val labels folder does not match"
        )

    def test_road_dataset_prep(self):
        yolov5 = YoloV5()
        yolov5.prepare_dataset("road")
        self.assertEqual(
            61144,
            len(os.listdir(f"{self.datasets_folder}/road/train/images/")),
            "Length of train images folder does not match"
        )
        self.assertEqual(
            61144,
            len(os.listdir(f"{self.datasets_folder}/road/train/labels/")),
            "Length of train labels folder does not match"
        )
        self.assertEqual(
            20381,
            len(os.listdir(f"{self.datasets_folder}/road/test/images/")),
            "Length of test images folder does not match"
        )
        self.assertEqual(
            20381,
            len(os.listdir(f"{self.datasets_folder}/road/test/labels/")),
            "Length of test labels folder does not match"
        )