from data.datasets import DatasetLoader
from data.datasets.data_augmentation import DataAugmentation
from data.models.yolov5 import YoloV5

dataset_loader = DatasetLoader()
gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
yoloV5 = YoloV5()
data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
data_augment_path = data_augmentation.blur_set(radius=4, frac=1)
yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)

