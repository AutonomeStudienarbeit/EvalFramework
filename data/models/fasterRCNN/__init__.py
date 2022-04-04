import torch.utils.data
import torchvision.models.detection
import torch.multiprocessing
import wandb

import utils
from data.models.fasterRCNN.torchDataset import TorchDataset
from data.models.fasterRCNN.dependencys.utils import collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.models.fasterRCNN.dependencys.engine import train_one_epoch, evaluate
from os import path as os_path, getcwd
from os.path import exists
from datetime import datetime


class FasterRCNN():

    def __init__(self, dataset):
        self.__location__ = os_path.realpath(os_path.join(getcwd(), os_path.dirname(__file__)))
        self.model = None
        self.torch_dataset = None
        self.dataset_loader = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(True)
        self.num_classes = dataset.number_of_classes + 1
        self.dataset = dataset
        self.optimizer = None
        self.device = None
        self.lr_scheduler = None
        self.wandb_logged_in = False
        self.backbone = "Default"
        self.model_setup()

    def model_setup(self):
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def set_backbone(self, backbone_name, is_pretrained):
        backbone_switch = {
            "resNet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
            "mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn
        }
        self.model = backbone_switch.get(backbone_name)(is_pretrained)
        self.backbone = backbone_name
        self.model_setup()

    def prepare_dataset(self, subset_name):
        self.torch_dataset = TorchDataset(self.dataset, subset_name=subset_name)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.num_classes = dataset.number_of_classes
        self.model_setup()

    def train(self, batch_size, num_epochs, print_freq):
        if self.dataset.dataset_id == "GTSRB": torch.multiprocessing.set_sharing_strategy('file_system')
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        if not self.wandb_logged_in:
            wandb.login
            wandb.init(project='faster-r-cnn', name=f'faster_r-cnn_resnet50_fpn_train_{self.dataset.dataset_id}_{num_epochs}Epochs',
                   config={"learning_rate": 0.005, "architecture": "CNN", "epochs": num_epochs})
            self.wandb_logged_in = True

        for epoch in range(num_epochs):
            losses = train_one_epoch(self.model, self.optimizer, self.dataset_loader, self.device, epoch,
                                     print_freq=print_freq)
            wandb.log({"loss": losses.meters.get('loss').median,
                       "loss_classifier": losses.meters.get('loss_classifier').median,
                       "loss_box_reg": losses.meters.get('loss_box_reg').median,
                       "loss_objectness": losses.meters.get('loss_objectness').median,
                       "loss_rpn_box_reg": losses.meters.get('loss_rpn_box_reg').median})
            self.lr_scheduler.step()
            self.validate(batch_size=batch_size, dataset=self.dataset, subset_name="test")
        save()

    def validate(self, batch_size, dataset=None, subset_name=None):
        if dataset.dataset_id == "GTSRB": torch.multiprocessing.set_sharing_strategy('file_system')

        if not self.wandb_logged_in:
            wandb.login
            wandb.init(project='faster-r-cnn', name=f'faster_r-cnn_resnet50_fpn_validate_{dataset.dataset_id}',
                   config={"learning_rate": 0.005, "architecture": "CNN", "epochs": num_epochs})
            self.wandb_logged_in = True

        if dataset is not None and subset_name is not None:
            tmp_torch_dataset = TorchDataset(dataset=dataset, subset_name=subset_name)
        else:
            tmp_torch_dataset = self.torch_dataset
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=tmp_torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        evaluation = evaluate(self.model, self.dataset_loader, self.device)
        evaluation_values = evaluation.coco_eval.get('bbox').stats
        wandb.log({"AP_IoU=0.50:0.95": evaluation_values[0],
                   "AP_IoU=0.50": evaluation_values[1],
                   "AP_IoU=0.75": evaluation_values[2],
                   "AR_IoU=0.50:0.95_maxDets=1": evaluation_values[6],
                   "AR_IoU=0.50:0.95_maxDets=10": evaluation_values[7],
                   "AR_IoU=0.50:0.95_maxDets=100": evaluation_values[8]})

    def save(self):
        utils.create_nested_folders(f"{self.__location__}/out/")
        states = self.model.state_dict() # get weights and biases of current model
        path = f"{self.__location__}/out/fasterRCNN_{self.backbone}_{self.dataset.dataset_id}_{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.pt"
        torch.save(states, path)
        return path

    def load_model(self, path):
        if not exists(path):
            raise Exception(f"File: {path} does not exist")
        filename = path.split("/")[-1]
        backbone = filename.split("_")[1]
        self.set_backbone(backbone_name=backbone, is_pretrained=True)
        self.model.load_state_dict(torch.load(path))
        self.model_setup()
