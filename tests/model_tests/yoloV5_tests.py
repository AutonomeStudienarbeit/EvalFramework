from data.models.yoloV5_torch.yoloV5 import YoloV5
from PIL import Image
from requests import get

yoloV5 = YoloV5()
image = cat_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
cat_image = Image.open(get(cat_url, stream=True).raw)
print(yoloV5.run_inference(cat_image))
