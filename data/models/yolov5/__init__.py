import os
import pandas as pd


class YoloV5:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def prepare_dataset(self, dataset_name):
        dataset_funcs = {"gtsrb": self._prepare_gtsrb}
        dataset_funcs.get(dataset_name)()

    def _prepare_gtsrb(self):
        # collect gtsrb file paths in txt file, so that yolo can read the dataset
        gtsrb_root = self.__location__ + "/../../datasets/gtsrb/"
        gtsrb_train = gtsrb_root + "train/"
        gtsrb_test = gtsrb_root + "test/"
        train_set = []
        test_set = []

        for class_folder in os.listdir(gtsrb_train):
            for file in os.listdir(gtsrb_train + f"{class_folder}/"):
                train_set.append(f"train/{class_folder}/{file}")

        for file in os.listdir(gtsrb_test):
            test_set.append(f"test/{file}")

        self._write_list_to_file(train_set, f"{gtsrb_root}train_paths.txt")
        self._write_list_to_file(test_set, f"{gtsrb_root}test_paths.txt")

        # convert gtsrb csv Labels to YoloFileFormat
        # YOLO format:
        # one *.txt file per image; The *.txt file specifications are:
        # - one row per object
        # - each row is [class x_center y_center width height] format
        # - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width
        #   by image width, and y_center and height by image height
        # - Class numbers are zero-indexed (start from 0)

        train_df = pd.read_csv(f"{gtsrb_root}Train.csv")
        test_df = pd.read_csv(f"{gtsrb_root}Test.csv")


    def _write_list_to_file(self, list, path):
        f = open(path, 'w+')
        for element in list:
            f.write(element + "\n")
        f.close()
