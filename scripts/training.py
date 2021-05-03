import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
from pathlib import Path
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.utils import make_grid
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tqdm import tqdm
from skimage import io
import random
from efficientnet_pytorch import EfficientNet
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
import logging
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.reproducibility import set_all_seeds
from src.dataset import GroceryDataset
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

imagenet_means = [0.485, 0.456, 0.406]
imagenet_stds = [0.229, 0.224, 0.225]
pretrained_img_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset() -> Tuple[pd.DataFrame, Dict]:
    """
    Returns the Freiburg dataset as a DataFrame and Label Map mapping from label to index

    Returns:
        Tuple[pd.DataFrame, Dict]: A tuple of the DataFrame for Freiburg dataset and Label mapping
        from the label name to an integer
    """

    data_dir = Path.cwd()/"freiburg_grocery_images"
    labels = [directory.name for directory in data_dir.iterdir()]
    label_map = {label: i for i, label in enumerate(labels)}

    all_items = [str(file) for label in labels for file in (data_dir/label).iterdir()]
    labels_of_items = [label for label in labels for file in (data_dir/label).iterdir()]

    df = pd.DataFrame({"Image": all_items, "Label": labels_of_items})
    return df, label_map

data_transforms = {
    "train": T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
        T.RandomCrop(pretrained_img_size),
        T.ToTensor(),
        T.Normalize(imagenet_means, imagenet_stds),

    ]),
    "valid": T.Compose([
        T.ToPILImage(),
        T.Resize(pretrained_img_size),
        T.ToTensor(),
        T.Normalize(imagenet_means, imagenet_stds),
    ])
}


def train(epochs, model, loss_func, optimizer, train_dl, valid_dl,):
    """
    Trains the model for a set number of epochs and evaluates the model at the end
    of every epoch
    """

    model = model.to(device)
    loop1 = tqdm(range(epochs), leave=True)
    epoch_loss = 0
    epoch_val_acc = 0
    for epoch in loop1:
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
                loop2 = tqdm(train_dl, total=len(train_dl), leave=False)
                running_loss = 0
                for inputs, labels in loop2:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loop2.set_description(f"Training: Epoch [{epoch+1}/{epochs}]")
                    epoch_loss = running_loss / len(train_dl)
            elif phase == "valid":
                model.eval()
                loop3 = tqdm(valid_dl, total=len(valid_dl), leave=False)
                total = 0
                correct = 0
                with torch.no_grad():
                    for inputs, labels in loop3:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        preds = torch.argmax(outputs, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()
                        loop3.set_description(f"Evaluation: Epoch [{epoch+1}/{epochs}]")
                    epoch_val_acc = correct/total
        loop1.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop1.set_postfix(val_acc=epoch_val_acc, training_loss=epoch_loss)

    print(epoch_val_acc)
    return epoch_val_acc


def init_resnet(num_classes: int) -> nn.Module:
    """
    Initialize a ResNet-50 model

    Args:
        num_classes (int): Number of Output Classes

    Returns:
        nn.Module: The ResNet model

    """
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

def init_efficientnet(num_classes: int) -> nn.Module:
    """
    Initialize an EfficientNet B1 model.

    Args:
        num_classes (int): Number of Output Classes

    Returns:
        nn.Module: The EfficientNet model
    """

    return EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes)

def main():

    parser = ArgumentParser()
    parser.add_argument("--model-name",
            help="Choose between a ResNet 50 and EfficientNet",
            default="efficientnet",
            choices=("resnet", "efficientnet")
            )
    parser.add_argument("--epochs",
            help="Number of epochs to train for",
            type=int,
            default=20,
            )
    parser.add_argument("--n-folds",
            help="Number of folds for Cross-Validation",
            type=int,
            default=5,
            )
    args = parser.parse_args()
    
    logger.info(f"Model chosen: {args.model_name}")
    logger.info(f"Number of Training Epochs: {args.epochs}")
    logger.info(f"Number of Folds for Cross-Validation: {args.n_folds}")



    set_all_seeds(2021)
    df, label_map = prepare_dataset()

    model_map = {
            "efficientnet": init_efficientnet,
            "resnet": init_resnet
            }

    skf = StratifiedKFold(n_splits=args.n_folds)
    accs = []

    for train_indices, valid_indices in skf.split(df["Image"], df["Label"]):
        train_df = df.iloc[train_indices]
        valid_df = df.iloc[valid_indices]
        train_df = train_df.reset_index()
        valid_df = valid_df.reset_index()

        train_ds = GroceryDataset(train_df, label_map, data_transforms["train"])
        valid_ds = GroceryDataset(valid_df, label_map, data_transforms["valid"])
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        valid_dl = DataLoader(valid_ds, batch_size=32, num_workers=4)
        epochs = args.epochs

        model = model_map[args.model_name](num_classes=len(label_map))

        loss_func = nn.CrossEntropyLoss()
        loss_func.to(device)

        optimizer = optim.Adam(model.parameters())

        acc = train(epochs, model, loss_func, optimizer, train_dl, valid_dl)
        accs.append(acc)

    kfold_accs = np.asarray(accs)

    logger.info(f"KFold Mean Accuracy: {kfold_accs.mean()}")
    logger.info(f"KFold Accuracy Standard Deviation: {kfold_accs.std()}")

    torch.save(model.state_dict, f"{args.model_name}_last_fold_checkpoint.pt")

if __name__ == "__main__":
    main()
