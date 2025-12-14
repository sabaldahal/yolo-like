import kagglehub
import shutil
import random
from pathlib import Path


TRAIN_RATIO = 0.8
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)


path = kagglehub.dataset_download(
    "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
)

dataset_root = Path(path)
print("Dataset downloaded to:", dataset_root)


raw_data = dataset_root / "agri_data" / "data"

if not raw_data.exists():
    raise RuntimeError(f"Raw data directory not found: {raw_data}")

print("Raw data dir:", raw_data)


images = [f for f in raw_data.iterdir() if f.suffix.lower() in IMAGE_EXTS]

random.shuffle(images)
split_idx = int(len(images) * TRAIN_RATIO)

train_images = images[:split_idx]
test_images = images[split_idx:]

print(f"Train images: {len(train_images)}")
print(f"Test images:  {len(test_images)}")

# -----------------------------------------------------------
# 4. Create YOLO directory structure
# -----------------------------------------------------------
yolo_root = Path("./dataset")

def make_dirs(split):
    img_dir = yolo_root / split / "images"
    lbl_dir = yolo_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir

train_img_dir, train_lbl_dir = make_dirs("train")
test_img_dir, test_lbl_dir = make_dirs("test")


def copy_pair(img_path, img_dst, lbl_dst):
    shutil.copy2(img_path, img_dst / img_path.name)

    label_path = img_path.with_suffix(".txt")
    if label_path.exists():
        shutil.copy2(label_path, lbl_dst / label_path.name)
    else:
        # Empty label file is valid in YOLO
        (lbl_dst / label_path.name).touch()

for img in train_images:
    copy_pair(img, train_img_dir, train_lbl_dir)

for img in test_images:
    copy_pair(img, test_img_dir, test_lbl_dir)

print("\nYOLO dataset prepared at:", yolo_root)
