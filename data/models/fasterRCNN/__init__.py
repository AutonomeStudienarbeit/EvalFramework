

class FasterRCNN():

    def __init__(self):
        self.model = None

    def set_backbone(self):
        # TODO: set Backbone via Jumptable
        raise Exception

    def prepare_dataset(self, dataset):
        # TODO: convert images to png
        # TODO: include get_transforms
        # TODO: create torch dataset
        raise Exception

    def train(self):
        # TODO: train model
        raise Exception

    def validate(self):
        # TODO: validate
        raise Exception