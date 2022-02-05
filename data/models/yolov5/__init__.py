import os
import pandas as pd
import numpy as np

from shutil import move


class YoloV5:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def prepare_dataset(self, dataset_name):
        dataset_funcs = {"gtsrb": self._prepare_gtsrb}
        dataset_funcs.get(dataset_name)()

    def _prepare_gtsrb(self):
        # collect gtsrb file paths in txt file, so that yolo can read the dataset
        gtsrb_root = self.__location__ + "/../../datasets/gtsrb/"
        gtsrb_train = gtsrb_root + "Train/"
        gtsrb_test = gtsrb_root + "Test/"

        os.mkdir(f"{gtsrb_train}images/")
        for folder in os.listdir(gtsrb_train):
            if folder == "images":
                continue
            for image in os.listdir(f"{gtsrb_train}{folder}/"):
                move(f"{gtsrb_train}{folder}/{image}", f"{gtsrb_train}images/")
            os.rmdir(f"{gtsrb_train}{folder}")

        os.mkdir(f"{gtsrb_test}images/")
        for image in os.listdir(gtsrb_test):
            if image == "images":
                continue
            move(f"{gtsrb_test}{image}", f"{gtsrb_test}images/")

        # convert gtsrb csv Labels to YoloFileFormat
        # YOLO format:
        # one *.txt file per image; The *.txt file specifications are:
        # - one row per object
        # - each row is [class x_center y_center width height] format
        # - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width
        #   by image width, and y_center and height by image height
        # - Class numbers are zero-indexed (start from 0)

        os.mkdir(f"{gtsrb_train}labels/")
        os.mkdir(f"{gtsrb_test}labels/")

        # [train_df, test_df]
        dfs = [pd.read_csv(f"{gtsrb_root}Train.csv"), pd.read_csv(f"{gtsrb_root}Test.csv")]
        for subset_df in dfs:
            # Normalize Coordinates
            subset_df["Roi.X1"] /= subset_df["Width"]
            subset_df["Roi.X2"] /= subset_df["Width"]
            subset_df["Roi.Y1"] /= subset_df["Height"]
            subset_df["Roi.Y2"] /= subset_df["Height"]

            for index, row in subset_df.iterrows():
                try:
                    current_subset_id, _, current_image = row.loc["Path"].split("/")
                except ValueError:
                    current_subset_id, current_image = row.loc["Path"].split("/")

                # convert to YoloFormat
                converted = np.array([
                    row.loc["ClassId"],
                    (row.loc["Roi.X1"] + row.loc["Roi.X2"]) / 2.0,
                    (row.loc["Roi.Y1"] + row.loc["Roi.Y2"]) / 2.0,
                    row.loc["Roi.X1"] + row.loc["Roi.X2"],
                    row.loc["Roi.Y1"] + row.loc["Roi.Y2"]
                ])

                # write to label file
                with open(f"{gtsrb_root}{current_subset_id}/labels/{current_image[:-4]}.txt", "w+") as f:
                    f.write(f"{int(converted[0])} {' '.join(map(str, converted[1:]))}")
