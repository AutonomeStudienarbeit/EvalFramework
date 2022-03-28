import utils
import numpy as np
import cv2


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

    def blur_set(self, radius, frac):
        from PIL import Image, ImageFilter
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/blur"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            with Image.open(image) as pil_image:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
                pil_image.save(f"{folder_path}/{image.split('/')[-1][:-4]}.png")

    def perturb_gaussian_noise(self, frac):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/gaussian_noise"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            gaussian = np.random.normal(0, 1, cv_image.size)
            gaussian = gaussian.reshape(cv_image.shape[0], cv_image.shape[1], cv_image.shape[2]).astype('uint8')
            perturbed_image = cv2.add(cv_image, gaussian)
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", perturbed_image)

