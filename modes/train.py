import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

from utils.architecture import *
from utils.dataset import *
from utils.loss import *
from utils.constants import *
from utils.helper import *
from utils.plot import *


# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    epoch_losses = {
        "total": 0.0,
        "box": 0.0,
        "object": 0.0,
        "no_object": 0.0,
        "class": 0.0,
    }

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(CONSTANTS.DEVICE)
        y0, y1, y2 = (
            y[0].to(CONSTANTS.DEVICE),
            y[1].to(CONSTANTS.DEVICE),
            y[2].to(CONSTANTS.DEVICE),
        )

        with torch.amp.autocast(CONSTANTS.DEVICE):
            # Getting the model predictions
            outputs = model(x)


            l0 = loss_fn(outputs[0], y0, scaled_anchors[0], return_components=True)
            l1 = loss_fn(outputs[1], y1, scaled_anchors[1], return_components=True)
            l2 = loss_fn(outputs[2], y2, scaled_anchors[2], return_components=True)

            # Total loss used for backward, calculated at each scale
            loss = (
                l0["total"]
                + l1["total"]
                + l2["total"]
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # Accumulate losses
        epoch_losses["total"] += loss.item()
        epoch_losses["box"] += (l0["box"] + l1["box"] + l2["box"]).item()
        epoch_losses["object"] += (l0["object"] + l1["object"] + l2["object"]).item()
        epoch_losses["no_object"] += (l0["no_object"] + l1["no_object"] + l2["no_object"]).item()
        epoch_losses["class"] += (l0["class"] + l1["class"] + l2["class"]).item()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)
        progress_bar.set_postfix({
            "tot": f"{epoch_losses['total']:.3f}",
            "box": f"{epoch_losses['box']:.3f}",
            "obj": f"{epoch_losses['object']:.3f}",
            "noobj": f"{epoch_losses['no_object']:.3f}",
            "cls": f"{epoch_losses['class']:.3f}",
        })


    # Average per epoch
    for k in epoch_losses:
        epoch_losses[k] /= len(losses)
    
    return epoch_losses

def validation_loop(loader, model, loss_fn, scaled_anchors):
    model.eval()

    losses = []
    epoch_losses = {
        "total": 0.0,
        "box": 0.0,
        "object": 0.0,
        "no_object": 0.0,
        "class": 0.0,
    }

    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONSTANTS.DEVICE)
            y0, y1, y2 = (
                y[0].to(CONSTANTS.DEVICE),
                y[1].to(CONSTANTS.DEVICE),
                y[2].to(CONSTANTS.DEVICE),
            )

            with torch.amp.autocast(CONSTANTS.DEVICE):
                outputs = model(x)

                l0 = loss_fn(outputs[0], y0, scaled_anchors[0], return_components=True)
                l1 = loss_fn(outputs[1], y1, scaled_anchors[1], return_components=True)
                l2 = loss_fn(outputs[2], y2, scaled_anchors[2], return_components=True)

                loss = l0["total"] + l1["total"] + l2["total"]
            
            losses.append(loss.item())
            epoch_losses["total"] += loss.item()
            epoch_losses["box"] += (l0["box"] + l1["box"] + l2["box"]).item()
            epoch_losses["object"] += (l0["object"] + l1["object"] + l2["object"]).item()
            epoch_losses["no_object"] += (l0["no_object"] + l1["no_object"] + l2["no_object"]).item()
            epoch_losses["class"] += (l0["class"] + l1["class"] + l2["class"]).item()

    for k in epoch_losses:
        epoch_losses[k] /= len(losses)

    model.train()
    return epoch_losses


def main():
    # Transform for training
    train_transform = A.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            A.LongestMaxSize(max_size=CONSTANTS.IMAGE_SIZE),
            # Pad remaining areas with zeros
            A.PadIfNeeded(
                min_height=CONSTANTS.IMAGE_SIZE, min_width=CONSTANTS.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            # Random color jittering
            A.ColorJitter(
                brightness=0.5, contrast=0.5,
                saturation=0.5, hue=0.5, p=0.5
            ),
            # Flip the image horizontally
            A.HorizontalFlip(p=0.5),
            # Normalize the image
            A.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
            ),
            # Convert the image to PyTorch tensor
            ToTensorV2()
        ], 
        # Augmentation for bounding boxes
        bbox_params=A.BboxParams(
                        format="yolo", 
                        min_visibility=0.4, 
                        label_fields=[]
                    )
    )

    val_transform = A.Compose(
        [
            # Rescale an image so that maximum side is equal to image_size
            A.LongestMaxSize(max_size=CONSTANTS.IMAGE_SIZE),
            # Pad remaining areas with zeros
            A.PadIfNeeded(
                min_height=CONSTANTS.IMAGE_SIZE, min_width=CONSTANTS.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            # Normalize the image
            A.Normalize(
                mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255
            ),
            # Convert the image to PyTorch tensor
            ToTensorV2()
        ],
        # Augmentation for bounding boxes 
        bbox_params=A.BboxParams(
                        format="yolo", 
                        min_visibility=0.4, 
                        label_fields=[]
                    )
    )

    model = YOLOv3(num_classes=CONSTANTS.NUM_CLASSES).to(CONSTANTS.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = CONSTANTS.LEARNING_RATE)
    loss_fn = YOLOLoss()
    scaler = torch.amp.GradScaler(CONSTANTS.DEVICE)

    train_dataset = Dataset(
        image_dir=CONSTANTS.TRAIN_IMAGES_DIR,
        label_dir=CONSTANTS.TRAIN_LABELS_DIR,
        anchors=CONSTANTS.ANCHORS,
        image_size=CONSTANTS.IMAGE_SIZE,
        grid_sizes=CONSTANTS.GRID_SIZE,
        num_classes=CONSTANTS.NUM_CLASSES,
        transform=train_transform
    )

    val_dataset = Dataset(
        image_dir=CONSTANTS.VAL_IMAGES_DIR,
        label_dir=CONSTANTS.VAL_LABELS_DIR,
        anchors=CONSTANTS.ANCHORS,
        image_size=CONSTANTS.IMAGE_SIZE,
        grid_sizes=CONSTANTS.GRID_SIZE,
        num_classes=CONSTANTS.NUM_CLASSES,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = CONSTANTS.BATCH_SIZE,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = CONSTANTS.BATCH_SIZE,
        num_workers = 2,
        shuffle = True,
        pin_memory = True,
    )

    scaled_anchors = (
        torch.tensor(CONSTANTS.ANCHORS) * 
        torch.tensor(CONSTANTS.GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(CONSTANTS.DEVICE)


    print("Training YOLO model")
    print(f"Classes: {CONSTANTS.CLASS_LABELS}")
    print(f"Epochs: {CONSTANTS.EPOCHS}")
    print(f"Learning rate: {CONSTANTS.LEARNING_RATE}")
    print(f"Image size: {CONSTANTS.IMAGE_SIZE}")
    print(f"Batch size: {CONSTANTS.BATCH_SIZE}")

    history = {}
    train_history = []
    val_history = []
    
    for e in range(1, CONSTANTS.EPOCHS+1):
        print("Epoch:", e)
        all_train_losses = training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        all_val_losses = validation_loop(val_loader, model, loss_fn, scaled_anchors)
        train_history.append({
            "epoch": e,
            **{k: float(v) for k, v in all_train_losses.items()}
        })
        val_history.append({
            "epoch": e,
            **{k: float(v) for k, v in all_val_losses.items()}
        })

        # Saving the model
        if CONSTANTS.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")
    
    history["train"] = train_history
    history["val"] = val_history

    import json
    with open("loss_history.json", "w") as f:
        json.dump(history, f, indent=4)