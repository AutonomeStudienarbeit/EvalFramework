import os

import pandas as pd
import torch
import torch.utils.data
from PIL import Image
from os.path import exists

import data.models.fasterRCNN.dependencys.transforms as t
from utils import create_nested_folders
from shutil import move, copy2


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subset_name):
        self.dataset = dataset
        self.subset_name = subset_name
        self.image_root = f"{dataset.path}/fasterRCNN/{subset_name}/images"  # dataset root for non ppm images

        if self.subset_name == "train":
            self.transforms = self._get_transforms()
        else:
            self.transforms = None

        create_nested_folders(self.image_root)
        self._convert_images()

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(self.image_root)))
        if subset_name == "train":
            gt_path = f"{self.dataset.path}/{self.dataset.train_ground_truth}"
        elif subset_name == "test" and self.dataset.test_ground_truth is not None:
            gt_path = f"{self.dataset.path}/{self.dataset.test_ground_truth}"
        else:
            gt_path = f"{self.dataset.path}/{self.dataset.train_ground_truth}"

        if dataset.dataset_id == "GTSDB":
            self.annotation = pd.read_csv(gt_path, sep=";",
                                          names=["Filename", "X1.ROI", "Y1.ROI", "X2.ROI", "Y2.ROI", "classID"])
        else:
            self.annotation = pd.read_csv(gt_path)
            self.annotation = self.annotation.join(self.annotation["Path"].str.split("/", expand=True)).rename(
                columns={2: 'File'})

        self._remove_not_annotated_images()

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.image_root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        filename = self.imgs[idx][:-4]
        # get bounding boxes
        # x1/y1 -
        # |     |
        # - - x2/y2

        if self.dataset.dataset_id == "GTSDB":
            image_df = self.annotation.loc[self.annotation["Filename"] == f"{filename}.ppm"].copy()
            boxes = [[
                row["X1.ROI"],
                row["Y1.ROI"],
                row["X2.ROI"],
                row["Y2.ROI"]
            ] for idx, row in image_df.iterrows()]
            image_df["classID"] = image_df["classID"] + 1
            labels = list(image_df["classID"])
        else:
            image_df = self.annotation.loc[self.annotation['File'] == f"{filename}.png"].copy()
            boxes = [[
                row["Roi.X1"],
                row["Roi.Y1"],
                row["Roi.X2"],
                row["Roi.Y2"]
            ] for idx, row in image_df.iterrows()]
            image_df["ClassId"] = image_df["ClassId"] + 1
            labels = list(image_df["ClassId"])

        # for i in enumerate(self.annotation.itertuples()):
        #     if (i[1][1][:-4] == filename):
        #         xmin = np.min(i[1][2])
        #         xmax = np.max(i[1][4])
        #         ymin = np.min(i[1][3])
        #         ymax = np.max(i[1][5])
        #         label = i[1][6] + 1
        #         boxes.append([xmin, ymin, xmax, ymax])
        #         labels.append(label)
        num_boxes = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def _remove_not_annotated_images(self):

        if self.dataset.dataset_id == "GTSDB":
            annotated_files = set([x[:-4] for x in self.annotation.Filename.unique()])
        else:
            annotated_files = set([x[:-4] for x in self.annotation.File.unique()])

        files_without_annotations = set([x[:-4] for x in os.listdir(self.image_root)]) - annotated_files
        to_be_removed = files_without_annotations.intersection(set([x[:-4] for x in self.imgs]))
        for file in to_be_removed:
            self.imgs.remove(f"{file}.png")

    def _get_transforms(self):
        transforms = [t.ToTensor(), t.RandomHorizontalFlip(0.5)]
        # converts the image, a PIL image, into a PyTorch Tensor
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        return t.Compose(transforms)

    def _convert_images(self):
        subset_switch = {
            "train": self.dataset.load_train_subset,
            "test": self.dataset.load_test_subset,
            "val": self.dataset.load_validation_subset
        }

        subset = subset_switch.get(self.subset_name)()
        for image_path in subset[0]:
            if not exists(f"{self.image_root}/{image_path.split('/')[-1][:-4]}.png"):
                if image_path[-4:] != ".png":
                    with Image.open(image_path) as image:
                        image_filename = image_path.split("/")[-1]
                        image.save(f"{self.image_root}/{image_filename[:-4]}.png")
                        image.close()
                else:
                    copy2(image_path, f"{self.image_root}/")
