import utils


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

