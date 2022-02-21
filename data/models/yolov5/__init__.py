import os
import pandas as pd
import numpy as np
import json
import multiprocessing as mp

from shutil import move, copy2
from re import match as match_regex
from PIL import Image
from pathlib import Path


def create_nested_folders(*paths):
    for path in paths:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)


class YoloV5:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def prepare_dataset(self, dataset_name):
        dataset_funcs = {
            "gtsrb": self._prepare_gtsrb,
            "gtsdb": self._prepare_gtsdb,
            "road": self._prepare_road,
        }
        dataset_funcs.get(dataset_name)()

    def _prepare_gtsrb(self):
        # collect gtsrb file paths in txt file, so that yolo can read the dataset
        gtsrb_root = self.__location__ + "/../../datasets/gtsrb/"
        gtsrb_train = gtsrb_root + "Train/"
        gtsrb_test = gtsrb_root + "Test/"

        create_nested_folders(f"{gtsrb_train}yolo/images/")
        for folder in os.listdir(gtsrb_train):
            if folder == "images":
                continue
            for image in os.listdir(f"{gtsrb_train}{folder}/"):
                move(f"{gtsrb_train}{folder}/{image}", f"{gtsrb_train}yolo/images/")
            os.rmdir(f"{gtsrb_train}{folder}")

        create_nested_folders(f"{gtsrb_test}/yolo/images/")
        for image in os.listdir(gtsrb_test):
            if image == "images":
                continue
            move(f"{gtsrb_test}{image}", f"{gtsrb_test}yolo/images/")

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
                    row.loc["Roi.X2"] - row.loc["Roi.X1"] if row.loc["Roi.X2"] - row.loc["Roi.X1"] < 1.0 else 1.0,
                    row.loc["Roi.Y2"] - row.loc["Roi.Y1"] if row.loc["Roi.Y2"] - row.loc["Roi.Y1"] < 1.0 else 1.0
                ])

                # write to label file
                with open(f"{gtsrb_root}{current_subset_id}/yolo/labels/{current_image[:-4]}.txt", "w+") as f:
                    f.write(f"{int(converted[0])} {' '.join(map(str, converted[1:]))}")

    def _prepare_gtsdb(self):
        gtsdb_root = f"{self.__location__}/../../datasets/gtsdb"
        gtsdb_train = f"{gtsdb_root}/TrainIJCNN2013"
        gtsdb_test = f"{gtsdb_root}/TestIJCNN2013"

        gt_df = pd.read_csv(f"{gtsdb_train}/TrainIJCNN2013/gt.txt",
                            sep=";",
                            names=["Filename", "X1.ROI", "Y1.ROI", "X2.ROI", "Y2.ROI", "classID"]
                            )

        image_size = (1360, 800)  # (width, height)

        # Normalize Coordinates
        gt_df["X1.ROI"] /= image_size[0]
        gt_df["X2.ROI"] /= image_size[0]
        gt_df["Y1.ROI"] /= image_size[1]
        gt_df["Y2.ROI"] /= image_size[1]

        # load available image filenames
        filenames = [image for image in os.listdir(f"{gtsdb_train}/TrainIJCNN2013") if match_regex(".*([.]ppm)", image)]
        filenames = pd.DataFrame(filenames)

        # create train, val & test splits
        train, val, test = np.split(filenames.sample(frac=1, random_state=42),
                                    [int(.6 * len(filenames)),
                                     int(.8 * len(filenames))])  # train: 80%, val: 20%, test: 20%

        print(len(train))
        print(len(val))
        print(len(test))

        self._prepare_gtsdb_split(train, "train", gt_df)
        self._prepare_gtsdb_split(val, "val", gt_df)
        self._prepare_gtsdb_split(test, "test", gt_df)

    def _prepare_gtsdb_split(self, split_df, split_name, gt_df):
        gtsdb_root = f"{self.__location__}/../../datasets/gtsdb"
        gtsdb_train = f"{gtsdb_root}/TrainIJCNN2013"
        gtsdb_test = f"{gtsdb_root}/TestIJCNN2013"

        create_nested_folders(
            f"{gtsdb_root}/{split_name}/images",
            f"{gtsdb_root}/{split_name}/labels",
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
            with Image.open(
                    f"{gtsdb_train}/TrainIJCNN2013/{image}") as im:  # Yolo can't read images in ppm format
                im.save(
                    f"{gtsdb_root}/yolo/{split_name}/images/{image[:-4]}.jpg")  # Therefore instead of moving the image, the image is copied and converted simultaneously
            with open(f"{gtsdb_root}/yolo/{split_name}/labels/{image[:-4]}.txt", "w+") as f:
                image_df = gt_df.loc[gt_df["Filename"] == image]
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

    def _prepare_road(self):
        road_root = f"{self.__location__}/../../datasets/road"
        with open(f"{road_root}/road_trainval_v1.0.json") as f:
            road_annots = json.load(f)

        frame_annotations = road_annots['db']

        # path, image_width, image_height, xmin, ymin, xmax, ymax, classId
        frame_annotations_cleaned = []
        for video in frame_annotations:
            for frame in frame_annotations[video]['frames']:
                frame_data = frame_annotations[video]['frames'][frame]
                if frame_data['annotated']:
                    for annotation in frame_data['annos']:
                        object_annotation = frame_data['annos'][annotation]
                        frame_annotations_cleaned.append(
                            [
                                f"{video}/{int(frame):05d}.jpg",
                                frame_data['width'],
                                frame_data['height'],
                                object_annotation['box'][0],
                                object_annotation['box'][1],
                                object_annotation['box'][2],
                                object_annotation['box'][3],
                                object_annotation['agent_ids'][0]
                            ]
                        )

        gt_df = pd.DataFrame(frame_annotations_cleaned,
                             columns=['path', 'image_width', 'image_height', 'xmin', 'ymin', 'xmax', 'ymax', 'classId'])

        road_all_frames = []
        for video in frame_annotations:
            road_all_frames += [f"{video}/{int(frame):05d}.jpg" for frame in frame_annotations[video]['frames']]

        road_all_frames = np.array(road_all_frames)
        road_all_frames_df = pd.DataFrame(road_all_frames)

        train, val, test = np.split(road_all_frames_df.sample(frac=1, random_state=42),
                                    [int(.6 * len(road_all_frames_df)),
                                     int(.8 * len(road_all_frames_df))])  # train: 80%, val: 20%, test: 20%

        self.mp_prepare_road(gt_df, (train, "train"), (val, "val"), (test, "test"))

    def mp_prepare_road(self, gt_df, *splits):
        thread_count = mp.cpu_count()
        road_root = f"{self.__location__}/../../datasets/road"

        for split_tuple in splits:

            split_name = split_tuple[1]
            split_df = split_tuple[0]

            create_nested_folders(
                f"{road_root}/yolo/{split_name}/images",
                f"{road_root}/yolo/{split_name}/labels",
            )

            split_frame_count = split_tuple[0].shape[0]
            thread_range = split_frame_count // thread_count

            split_df.reset_index(drop=True, inplace=True)

            processes = []
            for i in range(thread_count):
                if i == thread_count - 1:
                    split_subset_to_be_processed = split_df.iloc[i * thread_range:, :]
                else:
                    split_subset_to_be_processed = split_df.iloc[i * thread_range:(i + 1) * thread_range - 1, :]
                print(f"Process {i}, number of elements: {split_subset_to_be_processed.shape[0]}")
                processes.append(
                    mp.Process(
                        target=self._prepare_road_split_mp_kernel,
                        args=(split_subset_to_be_processed, split_name, gt_df)
                    )
                )

            for p in processes:
                p.start()

            for p in processes:
                p.join()

    def _prepare_road_split_mp_kernel(self, split_df, split_name, gt_df):
        road_root = f"{self.__location__}/../../datasets/road"

        # convert gtsdb csv Labels to YoloFileFormat
        # YOLO format:
        # one *.txt file per image; The *.txt file specifications are:
        # - one row per object
        # - each row is [class x_center y_center width height] format
        # - Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width
        #   by image width, and y_center and height by image height
        # - Class numbers are zero-indexed (start from 0)

        # ['path', 'image_width', 'image_height', 'xmin', 'ymin', 'xmax', 'ymax', 'classId']

        for image_path in split_df[0]:
            video, image = image_path.split("/")
            copy2(f"{road_root}/rgb-images/{image_path}", f"{road_root}/yolo/{split_name}/images/{video}-{image}")
            with open(f"{road_root}/yolo/{split_name}/labels/{video}-{image[:-4]}.txt", "w+") as f:
                image_df = gt_df.loc[gt_df["path"] == image_path]
                gt_converted = np.array([
                    [
                        row.loc['classId'],
                        (row.loc["xmin"] + row.loc["xmax"]) / 2.0,
                        (row.loc["ymin"] + row.loc["ymax"]) / 2.0,
                        row.loc["xmax"] - row.loc["xmin"],
                        row.loc["ymax"] - row.loc["ymin"]
                    ] for index, row in image_df.iterrows()])
                lines = [f"{int(entry[0])} {' '.join(map(str, entry[1:]))}" for entry in gt_converted]
                f.write('\n'.join(lines))
                f.close()
