import torch.utils.data

from data.models.fasterRCNN.torchDataset import TorchDataset
from data.models.fasterRCNN.dependencys.utils import collate_fn

class FasterRCNN():

    def __init__(self):
        self.model = None
        self.torch_dataset = None
        self.dataset_loader = None

    def set_backbone(self):
        # TODO: set Backbone via Jumptable
        raise Exception

    def prepare_dataset(self, dataset, subset_name):
        self.torch_dataset = TorchDataset(dataset, subset_name=subset_name)

    def train(self, batch_size):
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        # TODO: train model
        raise Exception

    def validate(self, batch_size):
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )

        # TODO: validate
        raise Exception