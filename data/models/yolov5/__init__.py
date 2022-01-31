import os


class YoloV5:

    def __init__(self):
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    def prepare_dataset(self, dataset_name):
        dataset_funcs = {"gtsrb": self._prepare_gtsrb}
        dataset_funcs.get(dataset_name)()

    def _prepare_gtsrb(self):
        gtsrb_root = self.__location__ + "/../../datasets/gtsrb/"
        gtsrb_train = gtsrb_root + "train/"
        gtsrb_test = gtsrb_root + "test/"
        train_set = []
        test_set = []

        for class_folder in os.listdir(gtsrb_train):
            for file in os.listdir(gtsrb_train + f"{class_folder}/"):
                train_set.append(f"train/{class_folder}/{file}")

        for file in os.listdir(gtsrb_test):
            test_set.append(f"test/{file}")

        self._write_list_to_file(train_set, f"{gtsrb_root}train_paths.txt")
        self._write_list_to_file(test_set, f"{gtsrb_root}test_paths.txt")

    def _write_list_to_file(self, list, path):
        f = open(path, 'w+')
        for element in list:
            f.write(element + "\n")
        f.close()
