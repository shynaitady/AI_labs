import os
import shutil
from pathlib import Path
from PIL import Image

import tensorflow as tf

layers = tf.keras.layers


RAW_DATASET_DIR = "dataset"
CLEAN_DATASET_DIR = "dataset_clean"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 20
USE_AUGMENTATION = False
MIN_IMAGES_PER_CLASS = 100

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# CLEAN DATASET


def clean_dataset():
    raw_path = Path(RAW_DATASET_DIR)
    clean_path = Path(CLEAN_DATASET_DIR)

    if clean_path.exists():
        shutil.rmtree(clean_path)

    clean_path.mkdir(parents=True, exist_ok=True)

    print("\nCLEANING DATASET")
    print("-" * 40)

    for class_folder in raw_path.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name
        output_class_folder = clean_path / class_name
        output_class_folder.mkdir(parents=True, exist_ok=True)

        valid_count = 0
        skipped_count = 0

        for img_path in class_folder.iterdir():
            if img_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                skipped_count += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                output_path = output_class_folder / f"{class_name}_{valid_count}.jpg"
                img.save(output_path, "JPEG", quality=95)

                valid_count += 1

            except Exception as e:
                skipped_count += 1
                print(f"Skipped: {img_path.name} | Reason: {e}")

        print(f"{class_name}: {valid_count} cleaned, {skipped_count} skipped")

    print("-" * 40)


clean_dataset()

#load dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    CLEAN_DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    CLEAN_DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Number of classes:", num_classes)

# PREPROCESSING

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.1),
])

normalization_layer = layers.Rescaling(1.0 / 255)

def preprocess_train(image, label):
    image = normalization_layer(image)

    if USE_AUGMENTATION:
        image = data_augmentation(image, training=True)

    return image, label


def preprocess_test(image, label):
    image = normalization_layer(image)
    return image, label


train_dataset = train_dataset.map(preprocess_train)
test_dataset = test_dataset.map(preprocess_test)

train_dataset = train_dataset.shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# model


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# train dataset

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)


# evaluating part


test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nFinal test accuracy:", test_accuracy)
print("Final test loss:", test_loss)

model.save("fruit_model.keras")

print("\nModel saved as fruit_model.keras")