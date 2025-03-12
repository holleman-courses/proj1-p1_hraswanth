import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Fix OneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load dataset and model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test_images")

# Load trained model
model = load_model(MODEL_PATH)

# Initialize metrics
total, correct, TP, FP, FN, TN = 0, 0, 0, 0, 0, 0

# Process test images
for filename in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, filename)
    try:
        img = Image.open(img_path).convert("RGB").resize((28, 28))
        img_array = np.array(img, dtype="float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        pred_prob = model.predict(img_array)[0][0]
        pred_label = 1 if pred_prob >= 0.5 else 0

        # Determine true label from filename
        true_label = 1 if filename.startswith("Ts") else 0  # "Ts" = shoe, "Tns" = non-shoe

        # Update metrics
        total += 1
        if pred_label == true_label:
            correct += 1
            if pred_label == 1: TP += 1
            else: TN += 1
        else:
            if pred_label == 1: FP += 1
            else: FN += 1

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Compute accuracy, precision, recall
accuracy = correct / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Print evaluation results
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
