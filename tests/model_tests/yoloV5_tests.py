from unittest import TestCase
from os import path, getcwd, listdir
from data.models.yolov5 import YoloV5

class YoloV5Tests(TestCase):
    datasets_folder = path.realpath(getcwd() + "/../../data/datasets/")

    def test_gtsrb_dataset_prep(self):
        yolov5 = YoloV5()
        yolov5.prepare_dataset("gtsrb")
        self.assertTrue("train_paths.txt" in listdir(f"{self.datasets_folder}/gtsrb/"))
