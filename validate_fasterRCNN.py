from data.datasets import DatasetLoader
from data.models.fasterRCNN import FasterRCNN
from data.datasets.data_augmentation import DataAugmentation
import wandb
import numpy as np

loader = DatasetLoader()
models_to_run = ['fasterRCNN-mobilenet_v3_large_320_fpn-COCO-transfer-GTSDB-07_04_2022_23:33:25.pt',
                 'fasterRCNN-resNet50_fpn-COCO-transfer-GTSDB-10_04_2022_14:11:12.pt']

for model in models_to_run:

    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: {model} val on unaugmented")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] inializing Model")
    fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
    print("[+] Model initialized")
    print("[*] Preparing Dataset for Model")
    fasterRCNN.prepare_dataset(subset_name="val")
    print("[+] Preparing Dataset for Model finished")
    print("[*] Loading saved model")
    fasterRCNN.load_model(
        f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
    print("[+] Model loaded")
    print("[*] starting validation")
    fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name='reference')
    wandb.finish()

    for i in range(1, 11):
        print(
            "\n-----------------------------------------------------------------------------------------------------------------\n")
        print(f"Running Experiment for: {model} val on Blur with radius = {i}")
        print("[*] loading Dataset")
        gtsdb_dataset = loader.load_dataset("GTSDB")
        print("[+] Dataset loaded")
        print("[*] inializing Model")
        fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
        print("[+] Model initialized")
        print("[*] Preparing Dataset for Model")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
        augmented_data_path = data_augmentation.blur_set(radius=i, frac=1)
        gtsdb_dataset.set_augmentated_set(augmented_data_path)
        fasterRCNN.prepare_dataset(subset_name="val")
        print("[+] Preparing Dataset for Model finished")
        print("[*] Loading saved model")
        fasterRCNN.load_model(
            f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
        print("[+] Model loaded")
        print("[*] starting validation")
        fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name=f'{i}_blur')
        wandb.finish()

    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: {model} val on gaussian noise")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] inializing Model")
    fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
    print("[+] Model initialized")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    augmented_data_path = data_augmentation.perturb_set_gaussian_noise(1)
    gtsdb_dataset.set_augmentated_set(augmented_data_path)
    fasterRCNN.prepare_dataset(subset_name="val")
    print("[+] Preparing Dataset for Model finished")
    print("[*] Loading saved model")
    fasterRCNN.load_model(
        f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
    print("[+] Model loaded")
    print("[*] starting validation")
    fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name='gaussian_noise')
    wandb.finish()

    for i in np.linspace(0, 1, 10, endpoint=False)[1:]:
        print(
            "\n-----------------------------------------------------------------------------------------------------------------\n")
        print(f"Running Experiment for: {model} val on {i} salt & pepper")
        print("[*] loading Dataset")
        gtsdb_dataset = loader.load_dataset("GTSDB")
        print("[+] Dataset loaded")
        print("[*] inializing Model")
        fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
        print("[+] Model initialized")
        print("[*] Preparing Dataset for Model")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
        augmented_data_path = data_augmentation.perturb_set_salt_pepper(1, i)
        gtsdb_dataset.set_augmentated_set(augmented_data_path)
        fasterRCNN.prepare_dataset(subset_name="val")
        print("[+] Preparing Dataset for Model finished")
        print("[*] Loading saved model")
        fasterRCNN.load_model(
            f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
        print("[+] Model loaded")
        print("[*] starting validation")
        fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val',
                            augmentation_name=f'{i}_salt_pepper')
        wandb.finish()

    for i in range(-200, 225, 25):
        if i == 0: continue
        print(
            "\n-----------------------------------------------------------------------------------------------------------------\n")
        print(f"Running Experiment for: {model} val on {i} brightness")
        print("[*] loading Dataset")
        gtsdb_dataset = loader.load_dataset("GTSDB")
        print("[+] Dataset loaded")
        print("[*] inializing Model")
        fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
        print("[+] Model initialized")
        print("[*] Preparing Dataset for Model")
        data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
        augmented_data_path = data_augmentation.perturb_set_image_brightness(brightness=i, frac=1)
        gtsdb_dataset.set_augmentated_set(augmented_data_path)
        fasterRCNN.prepare_dataset(subset_name="val")
        print("[+] Preparing Dataset for Model finished")
        print("[*] Loading saved model")
        fasterRCNN.load_model(
            f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
        print("[+] Model loaded")
        print("[*] starting validation")
        fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name=f'{i}_brightness')
        wandb.finish()

    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: {model} val on block")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] inializing Model")
    fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
    print("[+] Model initialized")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    augmented_data_path = data_augmentation.block_set(frac=1, color=(255, 255, 255))
    gtsdb_dataset.set_augmentated_set(augmented_data_path)
    fasterRCNN.prepare_dataset(subset_name="val")
    print("[+] Preparing Dataset for Model finished")
    print("[*] Loading saved model")
    fasterRCNN.load_model(
        f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
    print("[+] Model loaded")
    print("[*] starting validation")
    fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name='block')
    wandb.finish()

    print(
        "\n-----------------------------------------------------------------------------------------------------------------\n")
    print(f"Running Experiment for: {model} val on stickers")
    print("[*] loading Dataset")
    gtsdb_dataset = loader.load_dataset("GTSDB")
    print("[+] Dataset loaded")
    print("[*] inializing Model")
    fasterRCNN = FasterRCNN(dataset=gtsdb_dataset)
    print("[+] Model initialized")
    print("[*] Preparing Dataset for Model")
    data_augmentation = DataAugmentation(dataset=gtsdb_dataset, subset_to_be_perturbed="val")
    augmented_data_path = data_augmentation.add_stickers_to_set(1)
    gtsdb_dataset.set_augmentated_set(augmented_data_path)
    fasterRCNN.prepare_dataset(subset_name="val")
    print("[+] Preparing Dataset for Model finished")
    print("[*] Loading saved model")
    fasterRCNN.load_model(
        f"/home/leon/studienarbeit/EvalFramework/ref_models/{model}")
    print("[+] Model loaded")
    print("[*] starting validation")
    fasterRCNN.validate(batch_size=4, dataset=gtsdb_dataset, subset_name='val', augmentation_name='stickers')
    wandb.finish()
