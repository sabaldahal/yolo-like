import torch
import os


class CONSTANTS:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = False
    SAVE_MODEL = True
    CHECKPOINT_FILE = "checkpoint.pth.tar"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 30
    IMAGE_SIZE = 512
    GRID_SIZE = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    CLASS_LABELS = [
        "crop", "weed"
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], # Anchor boxes for each feature map scaled between 0 and 1
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # 3 feature maps at 3 different scales based on YOLOv3 paper
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    DATASET_DIR = './dataset'
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    TEST_DIR = os.path.join(DATASET_DIR, 'test')
    TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
    TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
    TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
    TEST_LABELS_DIR = os.path.join(TEST_DIR, 'labels')
    VAL_DIR = os.path.join(DATASET_DIR, 'val')
    VAL_IMAGES_DIR = os.path.join(VAL_DIR, 'images')
    VAL_LABELS_DIR = os.path.join(VAL_DIR, 'labels')

    @classmethod
    def recalculate_dataset_paths(cls,dataset_dir):
        cls.TRAIN_DIR = os.path.join(dataset_dir, 'train')
        cls.TEST_DIR = os.path.join(dataset_dir, 'test')
        cls.TRAIN_IMAGES_DIR = os.path.join(cls.TRAIN_DIR, 'images')
        cls.TRAIN_LABELS_DIR = os.path.join(cls.TRAIN_DIR, 'labels')
        cls.TEST_IMAGES_DIR = os.path.join(cls.TEST_DIR, 'images')
        cls.TEST_LABELS_DIR = os.path.join(cls.TEST_DIR, 'labels')
        cls.VAL_DIR = os.path.join(dataset_dir, 'val')
        cls.VAL_IMAGES_DIR = os.path.join(cls.VAL_DIR, 'images')
        cls.VAL_LABELS_DIR = os.path.join(cls.VAL_DIR, 'labels')