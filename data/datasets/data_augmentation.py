import os

import utils
from utils import inner_circles
from data.stickers import form_mappings
from re import match as match_regex
import numpy as np
import pandas as pd
import cv2


def draw_pixel(already_perturbed_pixels, rows, cols):
    new_pixel = False
    while not new_pixel:
        pixel = (
            np.random.randint(0, rows),
            np.random.randint(0, cols)
        )
        if pixel not in already_perturbed_pixels: new_pixel = True
    return pixel


class DataAugmentation:

    def __init__(self, dataset, subset_to_be_perturbed):
        self.dataset = dataset

        subset_switch = {
            "train": dataset.load_train_subset,
            "test": dataset.load_test_subset,
            "val": dataset.load_validation_subset
        }
        self.subset_name = subset_to_be_perturbed
        self.subset = subset_switch.get(subset_to_be_perturbed)()

        print("data auf subset:", self.subset)
        self.subset.to_csv("debug.csv")

    def blur_set(self, radius, frac):
        from PIL import Image, ImageFilter
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/blur"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            with Image.open(image) as pil_image:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
                pil_image.save(f"{folder_path}/{image.split('/')[-1][:-4]}.png")
        return folder_path

    def perturb_set_gaussian_noise(self, frac):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/gaussian_noise"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            gaussian = np.random.normal(0, 1, cv_image.size)
            gaussian = gaussian.reshape(cv_image.shape[0], cv_image.shape[1], cv_image.shape[2]).astype('uint8')
            perturbed_image = cv2.add(cv_image, gaussian)
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", perturbed_image)
        return folder_path

    def perturb_set_salt_pepper(self, frac_of_set, frac_of_images):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/salt_pepper"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac_of_set)
        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            row, col, _ = cv_image.shape
            already_perturbed_pixels = set()
            for i in range(int(row * col * frac_of_images)):
                pixel = draw_pixel(already_perturbed_pixels, row, col)
                already_perturbed_pixels.add(pixel)
                cv_image[pixel[0], pixel[1]] = [255, 255, 255] if (np.random.randint(0, 2)) else [0, 0, 0]
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)
        return folder_path

    def perturb_set_image_brightness(self, frac, brightness):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/image_brightness"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            cv_image = cv2.convertScaleAbs(cv_image, alpha=1, beta=brightness)
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)
        return folder_path

    def block_set(self, frac, color):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/block"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            drawn_pixels = set()
            cv_image = cv2.imread(image)
            row, col, _ = cv_image.shape
            start_pixel = draw_pixel(drawn_pixels, row, col)
            end_pixel = draw_pixel(drawn_pixels, row, col)
            # check if end_pixel out of range or in front of start_pixel
            if not end_pixel[0] > start_pixel[0]:
                tmp = end_pixel[0]
                end_pixel = (start_pixel[0], end_pixel[1])
                start_pixel = (tmp, start_pixel[1])
            if not end_pixel[1] > start_pixel[1]:
                tmp = end_pixel[1]
                end_pixel = (end_pixel[0], start_pixel[1])
                start_pixel = (start_pixel[0], tmp)
            if end_pixel[0] >= cv_image.shape[0]:
                end_pixel = (cv_image.shape[0] - 1, end_pixel[1])
            if end_pixel[1] >= cv_image.shape[0]:
                end_pixel = (end_pixel[0], cv_image.shape[1] - 1)

            rectangle = np.full((end_pixel[0] - start_pixel[0], end_pixel[1] - start_pixel[1], 3), color)
            cv_image[start_pixel[0]:end_pixel[0], start_pixel[1]:end_pixel[1]] = rectangle

            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)
            # print(f"Perturbed image {image.split('/')[-1][:-4]}, pixels: start{start_pixel}, end{end_pixel}")
        # print(f"Perturbed {len(subset_fraction[0])} images and got {edgeCase} edgeCases")
        return folder_path
        # TODO: Add a file which tracks the number of signs blocked by the randomly drawn rectangle

    def add_stickers_to_set(self, frac):
        form_to_func = {
            "triangle_south": inner_circles.calc_radius_triangle_south,
            "circle": inner_circles.get_radius_circle,
            "triangle_north": inner_circles.calc_radius_triangle_north,
            "stop_sign": inner_circles.get_radius_stop_sign,
            "rhombus": inner_circles.get_radius_rhombus
        }

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        sticker_repo_path = f"{__location__}/../stickers"
        stickers = np.array([f"{sticker_repo_path}/{sticker}" for sticker in os.listdir(sticker_repo_path) if
                             match_regex(".*([.](png))", sticker)])

        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/stickers"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        gt = pd.read_csv(f"{self.dataset.path}/{self.dataset.train_ground_truth}",
                         sep=";",
                         names=["Filename", "X1.ROI", "Y1.ROI", "X2.ROI", "Y2.ROI", "classID"]
                         )
        gt["ROI.HEIGHT"] = gt["Y2.ROI"] - gt["Y1.ROI"]
        gt["ROI.WIDTH"] = gt["X2.ROI"] - gt["X1.ROI"]

        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            image_gt = gt[gt["Filename"] == image.split("/")[-1]]
            for index, row in image_gt.iterrows():
                sticker_path = np.random.choice(stickers)
                sticker = cv2.imread(sticker_path)
                startpoint = (row["X1.ROI"], row["Y1.ROI"])
                endpoint = (row["X2.ROI"], row["Y2.ROI"])
                r, cp = form_to_func.get(
                    utils.get_key_by_value_of_list(form_mappings, row["classID"])
                )(startpoint, endpoint)
                # print(row)
                cv_image = inner_circles.add_sticker_in_circle(gt=row, image=cv_image, radius=r, sticker=sticker,
                                                               center=cp)
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)
        return folder_path

    #TODO: Add function that includes all augmentations into one folder
