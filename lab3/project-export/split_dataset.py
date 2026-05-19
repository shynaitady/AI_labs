import random
import shutil
from pathlib import Path

IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")
OUTPUT_DIR = Path("yolo_dataset")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

images = []
for ext in image_extensions:
    images.extend(IMAGES_DIR.glob(f"*{ext}"))

valid_images = []

for image_path in images:
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        valid_images.append(image_path)
    else:
        print(f"Skipped image without label: {image_path.name}")

random.shuffle(valid_images)

total = len(valid_images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": valid_images[:train_end],
    "val": valid_images[train_end:val_end],
    "test": valid_images[val_end:]
}

for split_name in ["train", "val", "test"]:
    (OUTPUT_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

for split_name, split_images in splits.items():
    for image_path in split_images:
        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        shutil.copy(image_path, OUTPUT_DIR / split_name / "images" / image_path.name)
        shutil.copy(label_path, OUTPUT_DIR / split_name / "labels" / label_path.name)

    print(f"{split_name}: {len(split_images)} images")

print("\nDataset split completed.")