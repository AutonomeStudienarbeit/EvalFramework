import os
import pandas as pd
import numpy as np

from shutil import move, copy2
from PIL import Image

from utils import create_nested_folders
import data.models.yolov5.yolov5_git.train as yolo_train
import data.models.yolov5.yolov5_git.val as yolo_val


class YoloV5:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def train(self, dataset, batch_size, weights, img_size, num_epochs, device=0):
        yml = f"{dataset.dataset_id.lower()}.yaml"
        yolo_train.run(data=yml, imgsz=img_size, weights=weights, device=device, batch_size=batch_size,
                       epochs=num_epochs)

    def val(self, dataset, batch_size, weights, img_size, device=0, task='val'):
        yml = f"{self.__location__}/yolov5_git/data/{dataset.dataset_id.lower()}.yaml"
        res = yolo_val.run(data=yml, batch_size=batch_size, imgsz=img_size, device=device, weights=weights, task=task)
        print(res)

    def prepare_dataset(self, dataset, is_data_augmentation=False, data_augmentation_path=None, task='val'):
        dataset_funcs = {
            "GTSRB": self._prepare_gtsrb,
            "GTSDB": self._prepare_gtsdb,
        }
        dataset_funcs.get(dataset.dataset_id)(dataset)
        if is_data_augmentation:
            self.copy_augmented_data(dataset, data_augmentation_path, task)

    def copy_augmented_data(self, dataset, data_augmentation_path, task):
        yolo_folder = f"{dataset.path}/yolo/{task}/images"
        for image in os.listdir(yolo_folder):
            os.remove(f"{yolo_folder}/{image}")

        for image in os.listdir(data_augmentation_path):
            copy2(f"{data_augmentation_path}/{image}", yolo_folder)

    def _prepare_gtsrb(self, dataset):
        # collect gtsrb file paths in txt file, so that yolo can read the dataset
        gtsrb_root = dataset.path
        gtsrb_train = gtsrb_root + "/Train/"
        gtsrb_test = gtsrb_root + "/Test/"

        create_nested_folders(f"{gtsrb_train}yolo/images/")
        for folder in os.listdir(gtsrb_train):
            if folder == "yolo":
                continue
            for image in os.listdir(f"{gtsrb_train}{folder}/"):
                copy2(f"{gtsrb_train}{folder}/{image}", f"{gtsrb_train}yolo/images/")

        create_nested_folders(f"{gtsrb_test}/yolo/images/")
        for image in os.listdir(gtsrb_test):
            if image == "yolo":
                continue
            copy2(f"{gtsrb_test}{image}", f"{gtsrb_test}yolo/images/")

        # convert gtsrb csv Labels to YoloFileFormat
        # YOLO format:
        # one *.txt file per image; The *.txt file specifications are:
        # - one row per object
        # - each row is [class x_center y_center width height] format
        # - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width
        #   by image width, and y_center and height by image height
        # - Class numbers are zero-indexed (start from 0)

        create_nested_folders(f"{gtsrb_train}yolo/labels/")
        create_nested_folders(f"{gtsrb_test}yolo/labels/")

        # [train_df, test_df]
        dfs = [pd.read_csv(f"{gtsrb_root}/Train.csv"), pd.read_csv(f"{gtsrb_root}/Test.csv")]
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
                    row.loc["Roi.X2"] - row.loc["Roi.X1"] if row.loc["Roi.X2"] - row.loc["Roi.X1"] < 1.0 else 1.0,
                    row.loc["Roi.Y2"] - row.loc["Roi.Y1"] if row.loc["Roi.Y2"] - row.loc["Roi.Y1"] < 1.0 else 1.0
                ])

                # write to label file
                with open(f"{gtsrb_root}/{current_subset_id}/yolo/labels/{current_image[:-4]}.txt", "w+") as f:
                    f.write(f"{int(converted[0])} {' '.join(map(str, converted[1:]))}")

    def _prepare_gtsdb(self, dataset):
        gtsdb_root = dataset.path

        gt_df = pd.read_csv(f"{gtsdb_root}/{dataset.train_ground_truth}",
                            sep=";",
                            names=["Filename", "X1.ROI", "Y1.ROI", "X2.ROI", "Y2.ROI", "classID"]
                            )

        image_size = (1360, 800)  # (width, height)

        # Normalize Coordinates
        gt_df["X1.ROI"] /= image_size[0]
        gt_df["X2.ROI"] /= image_size[0]
        gt_df["Y1.ROI"] /= image_size[1]
        gt_df["Y2.ROI"] /= image_size[1]

        train = dataset.load_train_subset()
        val = dataset.load_validation_subset()
        test = dataset.load_test_subset()

        print("prepared subset: ", val)

        print(len(train))
        print(len(val))
        print(len(test))

        self._prepare_gtsdb_split(train, "train", gt_df, dataset)
        self._prepare_gtsdb_split(val, "val", gt_df, dataset)
        self._prepare_gtsdb_split(test, "test", gt_df, dataset)

    def _prepare_gtsdb_split(self, split_df, split_name, gt_df, dataset):
        gtsdb_root = dataset.path

        create_nested_folders(
            f"{gtsdb_root}/yolo/{split_name}/images",
            f"{gtsdb_root}/yolo/{split_name}/labels",
        )

        # convert gtsdb csv Labels to YoloFileFormat
        # YOLO format:
        # one *.txt file per image; The *.txt file specifications are:
        # - one row per object
        # - each row is [class x_center y_center width height] format
        # - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width
        #   by image width, and y_center and height by image height
        # - Class numbers are zero-indexed (start from 0)

        for image in split_df[0]:
            with Image.open(image) as im:  # Yolo can't read images in ppm format
                image_file_name = image.split("/")[-1]
                im.save(
                    f"{gtsdb_root}/yolo/{split_name}/images/{image_file_name[:-4]}.png")  # Therefore instead of moving the image, the image is copied and converted simultaneously
            with open(f"{gtsdb_root}/yolo/{split_name}/labels/{image_file_name[:-4]}.txt", "w+") as f:
                image_df = gt_df.loc[gt_df["Filename"] == image_file_name]
                gt_converted = np.array([
                    [
                        row.loc['classID'],
                        (row.loc["X1.ROI"] + row.loc["X2.ROI"]) / 2.0,
                        (row.loc["Y1.ROI"] + row.loc["Y2.ROI"]) / 2.0,
                        row.loc["X2.ROI"] - row.loc["X1.ROI"],
                        row.loc["Y2.ROI"] - row.loc["Y1.ROI"]
                    ] for index, row in image_df.iterrows()])
                lines = [f"{int(entry[0])} {' '.join(map(str, entry[1:]))}" for entry in gt_converted]
                f.write('\n'.join(lines))
                f.close()
