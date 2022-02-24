from data.datasets import DatasetLoader

dataset_loader = DatasetLoader()
gtsrb_dataset = dataset_loader.load_dataset("GTSRB")

from data.models.yolov5 import YoloV5

yoloV5 = YoloV5()
yoloV5.prepare_dataset("gtsrb")
