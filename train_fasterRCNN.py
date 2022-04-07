from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN
import wandb

loader = DatasetLoader()
print("Running Experiment for: random init GTSDB resnet_50")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("resNet50_fpn", False)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=100)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: random init GTSRB resnet_50")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSRB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("resNet50_fpn", False)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=1000)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: COCO transfer GTSRB mobilenet")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSRB")
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
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=1000)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: random init GTSDB mobilenet")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("mobilenet_v3_large_320_fpn", False)
print("[+] Backbone Set")
print("[*] starting training")
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=100)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()

print("\n-----------------------------------------------------------------------------------------------------------------\n")
print("Running Experiment for: random init GTSRB mobilenet")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSRB")
print("[+] Dataset loaded")
print("[*] inializing Model")
fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
print("[+] Model initialized")
print("[*] Preparing Dataset for Model")
fasterRCNN.prepare_dataset(subset_name="train")
print("[+] Preparing Dataset for Model finished")
fasterRCNN.set_backbone("mobilenet_v3_large_320_fpn", False)
print("[+] Backbone Set")
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
fasterRCNN.train(batch_size=4, num_epochs=10, print_freq=100)
path = fasterRCNN.save()
print(f"[+] model saved to {path}")
wandb.finish()


