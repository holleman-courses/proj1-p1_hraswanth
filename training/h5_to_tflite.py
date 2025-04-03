import tensorflow as tf
import numpy as np
from PIL import Image
import os

# === Paths ===
model_path = "mobilenetv2_shoe_classifier.h5"
output_tflite_path = "model_float16.tflite"

# === Load model ===
model = tf.keras.models.load_model(model_path)

# === Convert to TFLite with float16 weights (safe + supported on Arduino with modifications)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# === Save the model ===
with open(output_tflite_path, "wb") as f:
    f.write(tflite_model)

print("✅ Float16 model saved to", output_tflite_path)