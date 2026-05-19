import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

layers = tf.keras.layers


DATASET_DIR = "dataset_clean"   
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

RESULTS_DIR = "model_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# DATASET

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.3,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# split temp_ds into validation and test
val_batches = tf.data.experimental.cardinality(temp_ds)
test_ds = temp_ds.take(val_batches // 2)
val_ds = temp_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.1),
])



def prepare_dataset(ds, preprocess_func, augment=False):
    def process(image, label):
        if augment:
            image = data_augmentation(image, training=True)
        image = preprocess_func(image)
        return image, label

    return ds.map(process).prefetch(AUTOTUNE)



def build_model(model_name, base_model_class, preprocess_func):
    base_model = base_model_class(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = preprocess_func(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=model_name)

    return model, base_model


def plot_history(history, model_name):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure()
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_loss.png")
    plt.close()



def evaluate_model(model, test_dataset, model_name):
    y_true = []
    y_pred = []

    start_time = time.time()

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    inference_time = time.time() - start_time

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nClassification report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        f"{RESULTS_DIR}/{model_name}_confusion_matrix.csv"
    )

    return precision, recall, f1, inference_time


def train_and_evaluate(model_name, base_model_class, preprocess_func, use_augmentation):
    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("Augmentation:", use_augmentation)
    print("=" * 60)

    model, base_model = build_model(model_name, base_model_class, preprocess_func)

    train_prepared = prepare_dataset(train_ds, lambda x: x, augment=use_augmentation)
    val_prepared = prepare_dataset(val_ds, lambda x: x, augment=False)
    test_prepared = prepare_dataset(test_ds, lambda x: x, augment=False)

    best_model_path = f"{RESULTS_DIR}/{model_name}_best.keras"
    last_model_path = f"{RESULTS_DIR}/{model_name}_last.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min"
        )
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_train_time = time.time()

    history_1 = model.fit(
        train_prepared,
        validation_data=val_prepared,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_2 = model.fit(
        train_prepared,
        validation_data=val_prepared,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    training_time = time.time() - start_train_time

    model.save(last_model_path)

  
    full_history = {}
    for key in history_1.history.keys():
        full_history[key] = history_1.history[key] + history_2.history[key]

    class HistoryObject:
        history = full_history

    plot_history(HistoryObject, model_name)

    val_loss, val_accuracy = model.evaluate(val_prepared, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_prepared, verbose=0)

    precision, recall, f1, inference_time = evaluate_model(model, test_prepared, model_name)

    total_params = model.count_params()
    trainable_params = np.sum([
        np.prod(v.shape) for v in model.trainable_weights
    ])

    result = {
        "Model": model_name,
        "Augmentation": use_augmentation,
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Training Time (s)": round(training_time, 2),
        "Inference Time (s)": round(inference_time, 4),
        "Validation Accuracy": round(val_accuracy, 4),
        "Test Accuracy": round(test_accuracy, 4),
        "Validation Loss": round(val_loss, 4),
        "Test Loss": round(test_loss, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1, 4)
    }

    return result



results = []

models_to_train = [
    (
        "VGG16_No_Augmentation",
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
        False
    ),
    (
        "VGG16_With_Augmentation",
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
        True
    ),
    (
        "ResNet50_With_Augmentation",
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet50.preprocess_input,
        True
    ),
    (
        "MobileNetV2_With_Augmentation",
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
        True
    ),
    (
        "EfficientNetB0_With_Augmentation",
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
        True
    )
]

for model_info in models_to_train:
    result = train_and_evaluate(*model_info)
    results.append(result)


results_df = pd.DataFrame(results)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)

print("\nMODEL COMPARISON TABLE")
print(results_df)

print(f"\nAll results saved in folder: {RESULTS_DIR}")