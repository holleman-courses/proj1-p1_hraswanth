import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# === Load model ===
model = load_model("model.h5")

# === Settings ===
img_dir = "dataset/test_images"
img_size = (224, 224)

# === Evaluation ===
correct = 0
total = 0

print("🔍 Evaluating test images:\n")
for file in sorted(os.listdir(img_dir)):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Determine true label based on filename
    fname = file.lower()
    if "tsh" in fname:
        true_label = 1
    elif "tns" in fname:
        true_label = 0
    else:
        true_label = "Unknown"

    # Load and preprocess image
    img_path = os.path.join(img_dir, file)
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_label = 1 if prediction >= 0.5 else 0
    confidence = prediction if predicted_label == 1 else 1 - prediction

    # Display result
    label_str = "Shoe" if predicted_label == 1 else "Non-shoe"
    true_str = (
        "Shoe" if true_label == 1 else "Non-shoe" if true_label == 0 else "Unknown"
    )
    correct_str = "✅" if true_label == predicted_label else "❌"

    print(
        f"{file:<12} ➝ {label_str:<9} ({confidence * 100:.2f}%) | True: {true_str:<9} {correct_str}"
    )

    if true_label != "Unknown":
        total += 1
        if true_label == predicted_label:
            correct += 1

# === Final Accuracy ===
if total > 0:
    acc = correct / total * 100
    print(f"\n✅ Accuracy: {correct}/{total} ({acc:.2f}%)")
else:
    print("\n⚠️ No labeled images found for accuracy calculation.")