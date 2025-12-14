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


def main():
    print("TEST IMAGES DIR:", CONSTANTS.TEST_IMAGES_DIR)
    # Setting the load_model to True
    load_model = True

    # Defining the model, optimizer, loss function and scaler
    model = YOLOv3(num_classes=CONSTANTS.NUM_CLASSES).to(CONSTANTS.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = CONSTANTS.LEARNING_RATE)
    loss_fn = YOLOLoss()
    scaler = torch.amp.GradScaler(CONSTANTS.DEVICE)

    # Loading the checkpoint
    if load_model:
        load_checkpoint(CONSTANTS.CHECKPOINT_FILE, model, optimizer, CONSTANTS.LEARNING_RATE)
    # Transform for testing
    test_transform = A.Compose(
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

    # Defining the test dataset and data loader
    test_dataset = Dataset(
        image_dir=CONSTANTS.TEST_IMAGES_DIR,
        label_dir=CONSTANTS.TEST_LABELS_DIR,
        anchors=CONSTANTS.ANCHORS,
        image_size=CONSTANTS.IMAGE_SIZE,
        grid_sizes=CONSTANTS.GRID_SIZE,
        num_classes=CONSTANTS.NUM_CLASSES,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 1,
        num_workers = 2,
        shuffle = True,
    )

    # Getting a sample image from the test data loader
    x, y = next(iter(test_loader))
    x = x.to(CONSTANTS.DEVICE)

    model.eval()
    with torch.no_grad():
        # Getting the model predictions
        output = model(x)
        # Getting the bounding boxes from the predictions
        bboxes = [[] for _ in range(x.shape[0])]
        anchors = (
                torch.tensor(CONSTANTS.ANCHORS)
                    * torch.tensor(CONSTANTS.GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                ).to(CONSTANTS.DEVICE)

        # Getting bounding boxes for each scale
        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(
                                output[i], anchor, s=S, is_predictions=True
                            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    model.train()

    # Plotting the image with bounding boxes for each image in the batch
    for i in range(batch_size):
        # Applying non-max suppression to remove overlapping bounding boxes
        nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6)
        # Plotting the image with bounding boxes
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)