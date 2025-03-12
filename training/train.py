import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, Input

# Fix OneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Load training data
X_train = []
y_train = []

for filename in os.listdir(os.path.join(DATASET_DIR, "shoe")):
    img = Image.open(os.path.join(DATASET_DIR, "shoe", filename)).convert("RGB")
    X_train.append(np.array(img, dtype="float32") / 255.0)
    y_train.append(1)

for filename in os.listdir(os.path.join(DATASET_DIR, "non_shoe")):
    img = Image.open(os.path.join(DATASET_DIR, "non_shoe", filename)).convert("RGB")
    X_train.append(np.array(img, dtype="float32") / 255.0)
    y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define CNN model with explicit Input layer
model = keras.Sequential([
    Input(shape=(28, 28, 3)),
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with proper epoch logging
model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=1)

# Save the trained model
model.save(MODEL_PATH)

print(f"Model training complete. Saved as '{MODEL_PATH}'.")