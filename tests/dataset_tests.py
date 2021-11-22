from data.datasets import DatasetLoader

dataset_loader = DatasetLoader()
gtsrb_dataset = dataset_loader.load_dataset("GTSRB")
print(gtsrb_dataset.folders)
print(gtsrb_dataset.csv_files)
train_subset = gtsrb_dataset.load_train_subset()
print(train_subset)