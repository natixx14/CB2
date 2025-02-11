CLASSES = {
        "zh-plus/tiny-imagenet": 200,
        "fashion_mnist": 10,
        "cub_200_2011": 200,
        "cifar-100": 100,
        "coco128": 80,
        "cifar-10": 10
}

TEST_SPLIT= {
        "zh-plus/tiny-imagenet": 'valid',
        "fashion_mnist": 'test',
        "cub_200_2011": "test",
        "cifar-100": "test",
        "cifar-10": "test",
        "coco128": "test"
}

DIM_EMBEDDINGS = {
        "openai/clip-vit-large-patch14": 768,
        "resnet18": 512,
        "vgg16": 512,
        "vitB16": 384,
        "yolo11": 3
        }

ID_LATENT = {
        "vitB16": -2,
        "vgg16": -3
        }
