available_datasets = {
    "GTSRB": {
        "description": "German Traffic Sign Recognition Benchmark",
        "url": "https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        "download_url": "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        "download_api": "kaggle",
        "required_data": "",
        "version": ""
    },
    "Cityscapes-Fine": {
        "description": "Cityscapes dataset with fine annotations",
        "url": "https://www.cityscapes-dataset.com/",
        "download_url": "gtFine_trainvaltest.zip",
        "download_api": "cityscapesscripts",
        "required_data":"Cityscapes-Fine-Images",
        "version": ""
    },
    "Cityscapes-Fine-Images": {
        "description": "Cityscapes dataset for fine annotations",
        "url": "https://www.cityscapes-dataset.com/",
        "download_url": "leftImg8bit_trainvaltest.zip",
        "download_api": "cityscapesscripts",
        "required_data":"Cityscapes-Fine",
        "version": ""
    }
}
