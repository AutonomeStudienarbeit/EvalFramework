from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN

loader = DatasetLoader()
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSRB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
# print("[*] Preparing Dataset for Model")
# fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("resNet50_fpn", True)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.validate(batch_size=2, dataset=gtsdb_dataset, subset_name='test')
