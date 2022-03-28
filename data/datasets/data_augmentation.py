import utils
import numpy as np
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

    def blur_set(self, radius, frac):
        from PIL import Image, ImageFilter
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/blur"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            with Image.open(image) as pil_image:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
                pil_image.save(f"{folder_path}/{image.split('/')[-1][:-4]}.png")

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

    def perturb_set_image_brightness(self, frac, brightness):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/image_brightness"
        utils.create_nested_folders(folder_path)

        subset_fraction = self.subset.sample(frac=frac)
        for image in subset_fraction[0]:
            cv_image = cv2.imread(image)
            cv_image = cv2.convertScaleAbs(cv_image, alpha=1, beta=brightness)
            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)

    def block_set(self, frac, color):
        folder_path = f"{self.dataset.path}/data-augmentation/{self.subset_name}/block"
        utils.create_nested_folders(folder_path)

        edgeCase = 0
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

            rectangle = np.full((end_pixel[0]-start_pixel[0], end_pixel[1]-start_pixel[1], 3), color)
            cv_image[start_pixel[0]:end_pixel[0], start_pixel[1]:end_pixel[1]] = rectangle

            cv2.imwrite(f"{folder_path}/{image.split('/')[-1][:-4]}.png", cv_image)
            # print(f"Perturbed image {image.split('/')[-1][:-4]}, pixels: start{start_pixel}, end{end_pixel}")
        # print(f"Perturbed {len(subset_fraction[0])} images and got {edgeCase} edgeCases")

        # TODO: Add a file which tracks the number of signs blocked by the randomly drawn rectangle
