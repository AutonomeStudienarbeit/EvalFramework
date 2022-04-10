from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN
import wandb

loader = DatasetLoader()

print("Running Experiment for: GTSRB transfer GTSDB resnet")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
print("[*] Loading saved model")
fasterRCNN.load_model(f"/home/leon/studienarbeit/EvalFramework/ref_models/fasterRCNN-resNet50_fpn-COCO-transfer-GTSRB-10_04_2022_19:35:54.pt")
print("[+] Model loaded")
print("[*] starting training")
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=1000)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: COCO transfer GTSDB mobilenet")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("mobilenet_v3_large_320_fpn", True)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.train(batch_size=4, num_epochs=100, print_freq=100)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: COCO transfer GTSDB resnet_50")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSRB")
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
fasterRCNN.train(batch_size=4, num_epochs=100, print_freq=1000)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

