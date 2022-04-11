from data.datasets import DatasetLoader
from data.datasets.data_augmentation import DataAugmentation
from data.models.yolov5 import YoloV5
import numpy as np

dataset_loader = DatasetLoader()

print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
print(f"Running Experiment for: val on unaugmented data")
gtsdb_dataset = dataset_loader.load_dataset("GTSDB")
yoloV5 = YoloV5()
# data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
# data_augment_path = data_augmentation.blur_set(radius=4, frac=1)
yoloV5.prepare_dataset(dataset=gtsdb_dataset)
yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

loader = DatasetLoader()

for i in range(1, 11):
    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: val on Blur with radius = {i}")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    data_augment_path = data_augmentation.blur_set(radius=i, frac=1)
    yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
    yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

print(
    "\n-----------------------------------------------------------------------------------------------------------------\n")
print(f"Running Experiment for: val on gaussian noise")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] Preparing Dataset for Model")
data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
data_augment_path = data_augmentation.perturb_set_gaussian_noise(1)
yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

for i in np.linspace(0, 1, 10, endpoint=False)[1:]:
    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: val on {i} salt & pepper")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    data_augment_path = data_augmentation.perturb_set_salt_pepper(1, i)
    yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
    yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

for i in range(-200, 225, 25):
    if i == 0: continue
    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: val on {i} brightness")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    data_augment_path = data_augmentation.perturb_set_image_brightness(brightness=i, frac=1)
    yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
    yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

print(
    "\n-----------------------------------------------------------------------------------------------------------------\n")
print(f"Running Experiment for: val on block")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] Preparing Dataset for Model")
data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
data_augment_path = data_augmentation.block_set(frac=1, color=(255, 255, 255))
yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)

print(
    "\n-----------------------------------------------------------------------------------------------------------------\n")
print(f"Running Experiment for: val on stickers")
print("[*] loading Dataset")
gtsdb_dataset = loader.load_dataset("GTSDB")
print("[+] Dataset loaded")
print("[*] Preparing Dataset for Model")
data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
data_augment_path = data_augmentation.add_stickers_to_set(1)
yoloV5.prepare_dataset(dataset=gtsdb_dataset, is_data_augmentation=True, data_augmentation_path=data_augment_path)
yoloV5.val(dataset=gtsdb_dataset, batch_size=2, weights='e100-yolov5m-gtsrb-gtsdb-best.pt', img_size=1360, device=1)