from data.datasets import DatasetLoader

dataset_loader = DatasetLoader()
gtsdb_dataset = dataset_loader.load_dataset("road")

from data.models.yolov5 import YoloV5

yoloV5 = YoloV5()
yoloV5.prepare_dataset('road')
