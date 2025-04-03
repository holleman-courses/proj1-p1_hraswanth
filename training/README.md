# Model Training – Shoe vs. Non-Shoe Classifier

This folder contains all training-related code, models, and evaluation scripts for a binary classifier that distinguishes between images of shoes and non-shoes using transfer learning.

---

## Folder Structure

```
training/
├── dataset/                 # Image data (train/validation subfolders)
├── models/                  # Saved .tflite and .h files
├── quant_samples/          # 224x224 RGB images used for quantization
├── mobilenetv2_shoe_classifier.h5     # Trained Keras model
├── model_float16.tflite     # Quantized TFLite model (float16)
├── model_data.h             # Header version of TFLite model for Arduino
├── train.py                 # Main training script
├── eval.py                  # Simple evaluation/inference script
├── h5_to_tflite.py          # Converts .h5 to TFLite (with quantization)
├── tflite_to_h.py           # Converts TFLite model to .h file via xxd
├── training_plot.png        # Accuracy/loss plot (output of training)
└── README.md                # This file
```

---

## Model Overview

- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224x3
- **Output**: Shoe (1) or Non-shoe (0)
- **Framework**: TensorFlow 2.5
- **Final Layer**: Dense(1, activation='sigmoid')

---

## Training Summary

- **Script**: `train.py`
- **Augmentation**:
  - Horizontal Flip
  - Rotation Range
  - Zoom Range
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Epochs**: 15
- **Validation Split**: 20%
- **Output**: `mobilenetv2_shoe_classifier.h5`

---

## Conversion Workflow

1. Convert `.h5` to quantized `.tflite` using:
   ```bash
   python h5_to_tflite.py
   ```
   Output: `model_float16.tflite`

2. Convert `.tflite` to `.h` for Arduino:
   ```bash
   python tflite_to_h.py
   ```
   Output: `model_data.h`

---

## Evaluation

Run `eval.py` to test inference on sample images:
```bash
python eval.py
```

It will print the model’s predicted probabilities for each input image.

---

## Notes

- Quantization used float16 for memory efficiency on microcontrollers.
- Model accuracy improved with basic augmentation and transfer learning.
- Training plot shows validation stabilizing around 83% accuracy.

---

## Author

Developed by **Sai Venkata Hraswanth Jangam**  
MS Computer Engineering @ UNC Charlotte  
Spring 2025 — ECGR 4127/5127: ML for IoT
