import torch.utils.data
import torchvision.models.detection
import wandb

from data.models.fasterRCNN.torchDataset import TorchDataset
from data.models.fasterRCNN.dependencys.utils import collate_fn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data.models.fasterRCNN.dependencys.engine import train_one_epoch, evaluate

class FasterRCNN():

    def __init__(self, dataset):
        self.model = None
        self.torch_dataset = None
        self.dataset_loader = None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(True)
        self.num_classes = dataset.number_of_classes + 1
        self.dataset = dataset
        self.optimizer = None
        self.device = None
        self.lr_scheduler = None
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
        self.model_setup()

    def prepare_dataset(self, subset_name):
        self.torch_dataset = TorchDataset(self.dataset, subset_name=subset_name)

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.num_classes = dataset.number_of_classes
        self.model_setup()

    def train(self, batch_size, num_epochs, print_freq):
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        wandb.login
        wandb.init(project='faster-r-cnn', name=f'faster_r-cnn_resnet50_fpn_{num_epochs}Epochs',
                   config={"learning_rate": 0.005, "architecture": "CNN", "epochs": num_epochs})
        for epoch in range(num_epochs):
            losses = train_one_epoch(self.model, self.optimizer, self.dataset_loader, self.device, epoch, print_freq=print_freq)
            wandb.log({"loss": losses.meters.get('loss').median,
                       "loss_classifier": losses.meters.get('loss_classifier').median,
                       "loss_box_reg": losses.meters.get('loss_box_reg').median,
                       "loss_objectness": losses.meters.get('loss_objectness').median,
                       "loss_rpn_box_reg": losses.meters.get('loss_rpn_box_reg').median})
            self.lr_scheduler.step()
            self.validate()

    def validate(self, batch_size):
        self.dataset_loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        evaluation = evaluate(self.model, self.data_loader, self.device)
        evaluation_values = evaluation.coco_eval.get('bbox').stats
        wandb.log({"AP_IoU=0.50:0.95": evaluation_values[0],
                   "AP_IoU=0.50": evaluation_values[1],
                   "AP_IoU=0.75": evaluation_values[2],
                   "AR_IoU=0.50:0.95_maxDets=1": evaluation_values[6],
                   "AR_IoU=0.50:0.95_maxDets=10": evaluation_values[7],
                   "AR_IoU=0.50:0.95_maxDets=100": evaluation_values[8]})