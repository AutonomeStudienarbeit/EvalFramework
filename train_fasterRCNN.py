from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN

loader = DatasetLoader()
gtsdb_dataset = loader.load_dataset("GTSDB")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
fasterRCNN.prepare_dataset(subset_name="train")
fasterRCNN.set_backbone("resNet50_fpn")
fasterRCNN.train(batch_size=2, num_epochs=10)