from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN

loader = DatasetLoader()
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("resNet50_fpn", True)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.train(batch_size=2, num_epochs=10, print_freq=10)
