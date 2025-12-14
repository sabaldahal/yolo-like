import torch
from utils.constants import *
from utils.dataset import *
from utils.helper import *
from utils.plot import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def main():
        
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


    # Creating a dataset object
    dataset = Dataset(
        image_dir=CONSTANTS.TEST_IMAGES_DIR,
        label_dir=CONSTANTS.TEST_LABELS_DIR,
        grid_sizes=CONSTANTS.GRID_SIZE,
        image_size=CONSTANTS.IMAGE_SIZE,
        num_classes=CONSTANTS.NUM_CLASSES,
        anchors=CONSTANTS.ANCHORS,
        transform=test_transform
    )

    # Creating a dataloader object
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

 
    scaled_anchors = torch.tensor(CONSTANTS.ANCHORS) / (
        1 / torch.tensor(CONSTANTS.GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    # Getting a batch from the dataloader
    x, y = next(iter(loader))

    # Getting the boxes coordinates from the labels
    # and converting them into bounding boxes without scaling
    boxes = []
    for i in range(y[0].shape[1]):
        anchor = scaled_anchors[i]
        boxes += convert_cells_to_bboxes(
                y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor
                )[0]

    # Applying non-maximum suppression
    boxes = nms(boxes, iou_threshold=1, threshold=0.7)

    # Plotting the image with the bounding boxes
    plot_image(x[0].permute(1,2,0).to("cpu"), boxes)