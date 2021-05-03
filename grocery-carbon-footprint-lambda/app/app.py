import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.models as models
import base64
import json
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
from PIL import Image
from io import BytesIO
from torchvision.models.resnet import ResNet
from typing import List
import pandas as pd

IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRETRAINED_IMG_SIZE = (224, 224)

LABELS = [
        'VINEGAR',
        'PASTA',
        'JUICE',
        'SODA',
        'CHOCOLATE',
        'CEREAL',
        'CORN',
        'CHIPS',
        'COFFEE',
        'MILK',
        'BEANS',
        'FISH',
        'TOMATO_SAUCE',
        'OIL',
        'CANDY',
        'CAKE',
        'FLOUR',
        'WATER',
        'SPICES',
        'TEA',
        'HONEY',
        'NUTS',
        'SUGAR',
        'JAM',
        'RICE'
        ]

class Model:
    def __init__(self, model: nn.Module, labels: List[str]):
        self.model = model
        self.labels = labels
        self.transforms = data_transforms = T.Compose([
                T.Resize(PRETRAINED_IMG_SIZE),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEANS, IMAGENET_STD)
                ])

    def predict(self, image: Image) -> str:
        logits = self.model(self.transforms(image).unsqueeze(dim=0))
        predicted_label_index = torch.argmax(logits).item()
        return self.labels[predicted_label_index]


def load_model(model_file='/opt/ml/model'):
    resnet_model = models.resnet50()
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, len(LABELS))
    resnet_model.load_state_dict(torch.load(model_file))
    resnet_model.eval()
    return Model(resnet_model, LABELS)

def load_data(data_file="carbon_footprint_grocery.csv"):
    df = pd.read_csv(data_file)
    return df

model = load_model()
df = load_data()

def lambda_handler(event, context):
    image_bytes = event['body'].encode("utf-8")
    bytes_content = BytesIO(base64.b64decode(image_bytes))
    image = Image.open(bytes_content)

    model_prediction = model.predict(image)

    if (model_prediction in df["Food"].values):

        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "predicted_label": model_prediction,
                    "Emissions": df.loc[df["Food"] == model_prediction]["Emissions (gCO2e)"].values.squeeze().item(),
                }
            )
        }
    else:

        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "predicted_label": model_prediction
                }
            )
        }

