# Shoe vs. Non-Shoe Classification on Arduino Nano 33 BLE

This project is part of **ECGR 4127/5127: Visual Object Detection**. The goal is to deploy a lightweight machine learning model on an Arduino Nano 33 BLE with an OV767X camera to classify real-time images as either "Shoe" or "Non-Shoe".

---

## Project Structure

```
.
├── embedded/         # PlatformIO project for Arduino deployment
│   ├── include/      # Contains model_data.h and headers
│   ├── src/          # main.cpp and logic to capture image & run inference
│   └── platformio.ini
├── training/         # Python-based training scripts and datasets
│   ├── model/        # Saved .h5, .tflite, .h model files
│   ├── quant_samples/ # Images used for quantization
│   └── train.py      # Transfer learning script using MobileNetV2
├── .gitignore
└── README.md         # This file
```

---

## Model Summary

- **Architecture**: Transfer Learning with MobileNetV2
- **Input Size**: 224x224x3 RGB
- **Output**: Binary classification - Shoe (1), Non-shoe (0)
- **Framework**: TensorFlow 2.5 (for Arduino compatibility)
- **Conversion**: `.h5` → quantized `.tflite` → `.h` (header) via xxd

---

## Dataset

- **Camera**: OV767X on Arduino Nano 33 BLE
- **Samples**:
  - 15 images of shoes
  - 15 images of non-shoes
- **Augmentation**: Rotation, zoom, flip, etc. to artificially expand dataset

---

## Training

- **Script**: `training/train.py`
- **Augmentation**: Enabled using `ImageDataGenerator`
- **Optimizer**: Adam
- **Epochs**: 15
- **Validation Split**: 20%

---

## Conversion to Embedded

1. Trained `.h5` model: `mobilenetv2_shoe_classifier.h5`
2. Quantized to `.tflite`: `model_quant.tflite`
3. Converted to `.h` file:
   ```bash
   xxd -i model_quant.tflite > model_data.h
   ```

---

## Embedded Deployment

- **Board**: Arduino Nano 33 BLE Sense Lite
- **Camera**: OV767X (Arduino_OV767X v0.0.2)
- **Libraries**:
  - `Arduino_OV767X`
  - `Chirale_TensorFlowLite`
  - `ArduTFLite`
- **Code**: Captures camera input and runs TFLite inference in `main.cpp`

---

## Inference

Model processes camera images in RGB565 format and prints classification results via Serial Monitor.

Example:
```
Shoe: 0.82
Non-shoe: 0.18
Prediction: Shoe
```

---

## Known Issues

- Model contains ops unsupported by older TFLite for Microcontrollers
- FlatBuffer version mismatches triggered compilation issues
- Image preprocessing for training was manual and tedious
- Model conversion to `.h` was done offline due to memory constraints

---

## Future Improvements

- Collect higher quality images with better lighting
- Increase training dataset size
- Use SD card or BLE to store/send classification results
- Try models optimized for embedded systems (e.g. MobileNetV1 SSD, CMSIS-NN)

---

## Author

Developed by **Sai Venkata Hraswanth Jangam**  
MS Computer Engineering @ UNC Charlotte  
Spring 2025 — ECGR 4127/5127: ML for IoT
