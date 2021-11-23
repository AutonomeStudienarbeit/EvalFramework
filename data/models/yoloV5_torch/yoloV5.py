import torch


class YoloV5:

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def run_inference(self, image):
        torch_results = self.model(image)
        return torch_results.pandas().xyxy[0]
