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
