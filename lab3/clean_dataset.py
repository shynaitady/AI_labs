import os
import hashlib
from pathlib import Path
from PIL import Image, ImageOps

SOURCE_DIR = Path("dataset/fruits")
OUTPUT_DIR = Path("cleaned_dataset/images")

TARGET_SIZE = (640, 640)
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

used_hashes = set()
counter = 1


def calculate_hash(file_path):
    with open(file_path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


for file_path in SOURCE_DIR.iterdir():
    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"Skipped unsupported file: {file_path.name}")
        continue

    try:
        file_hash = calculate_hash(file_path)

        if file_hash in used_hashes:
            print(f"Duplicate removed: {file_path.name}")
            continue

        used_hashes.add(file_hash)

        image = Image.open(file_path)
        image.verify()

        image = Image.open(file_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        if image.width < 300 or image.height < 300:
            print(f"Low-quality image removed: {file_path.name}")
            continue

        image = image.resize(TARGET_SIZE)

        new_filename = f"image_{counter:04d}.jpg"
        output_path = OUTPUT_DIR / new_filename

        image.save(output_path, "JPEG", quality=95)

        print(f"Saved: {new_filename}")
        counter += 1

    except Exception as error:
        print(f"Corrupted image removed: {file_path.name} | Error: {error}")

print(f"\nCleaning finished. Total images saved: {counter - 1}")