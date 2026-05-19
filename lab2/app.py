import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import filedialog

# CONFIG

MODEL_PATH = "model_results/EfficientNetB0_With_Augmentation_best.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "apple",
    "banana",
    "peaches",
    "pomegranate",
    "strawberry"
]

# LOAD MODEL

model = tf.keras.models.load_model(MODEL_PATH)

# FUNCTIONS

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.webp")
        ]
    )

    if not file_path:
        return

    image = Image.open(file_path).convert("RGB")
    image_resized = image.resize((300, 300))

    tk_image = ImageTk.PhotoImage(image_resized)

    image_label.config(image=tk_image)
    image_label.image = tk_image

    predict_image(file_path)


def predict_image(file_path):
    processed_image = preprocess_image(file_path)

    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    result_label.config(
        text=f"Predicted class: {predicted_class}\nConfidence: {confidence:.2f}%"
    )

    probabilities_text = "Class probabilities:\n"

    for class_name, probability in zip(CLASS_NAMES, predictions[0]):
        probabilities_text += f"{class_name}: {probability * 100:.2f}%\n"

    probabilities_label.config(text=probabilities_text)


root = tk.Tk()
root.title("Fruit Image Classification")
root.geometry("500x650")

title_label = tk.Label(
    root,
    text="Fruit Image Classification",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=15)

upload_button = tk.Button(
    root,
    text="Upload Image and Run Inference",
    command=upload_image,
    font=("Arial", 12),
    width=30
)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=15)

result_label = tk.Label(
    root,
    text="Predicted class: -\nConfidence: -",
    font=("Arial", 14)
)
result_label.pack(pady=10)

probabilities_label = tk.Label(
    root,
    text="Class probabilities:",
    font=("Arial", 11),
    justify="left"
)
probabilities_label.pack(pady=10)

root.mainloop()