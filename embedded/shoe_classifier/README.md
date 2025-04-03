# Model Training – Shoe vs. Non-Shoe Classification

This folder contains the code and resources required to train a binary image classification model for detecting whether an image contains a shoe or not. The final model is intended for deployment on the Arduino Nano 33 BLE Sense with an OV767X camera.

---

## Folder Structure

```
training/
├── model/             # Stores .h5, .tflite, and .h model formats
├── quant_samples/     # Sample images for post-training quantization
├── plots/             # Accuracy and loss graphs
├── train.py           # Transfer learning script using MobileNetV2
└── convert_to_h.py    # Optional utility for generating .h from .tflite
```

---

## Model Details

- Architecture: MobileNetV2 (transfer learning)
- Input Size: 224x224 RGB
- Output Classes: 
  - 1 = Shoe  
  - 0 = Non-shoe
- Framework: TensorFlow 2.5 (for compatibility with Arduino)
- Loss Function: Binary cross-entropy
- Optimizer: Adam
- Epochs: 15
- Validation Split: 0.2 (20%)

---

## Dataset

The training dataset was captured manually using the OV767X camera on the Arduino. Each frame was saved in .txt logs (RGB565), converted to .png, and labeled manually.

- Shoe images: 15  
- Non-shoe images: 15  
- Augmentation Techniques:  
  - Random flips  
  - Rotation  
  - Zoom  
  - Brightness adjustment  

Augmentation was applied using ImageDataGenerator during training.

---

## Training Instructions

1. Place training images into `data/shoe/` and `data/non_shoe/`.
2. Run `train.py`:
   ```bash
   python train.py
   ```
3. After training:
   - The `.h5` model will be saved as `mobilenetv2_shoe_classifier.h5`
   - Accuracy/loss curves will be saved in the `plots/` folder

---

## Quantization & Conversion

1. Convert `.h5` → `.tflite`:
   ```python
   import tensorflow as tf

   model = tf.keras.models.load_model("mobilenetv2_shoe_classifier.h5")
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = ...
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.uint8
   converter.inference_output_type = tf.uint8
   tflite_model = converter.convert()

   with open("model_quant.tflite", "wb") as f:
       f.write(tflite_model)
   ```

2. Convert `.tflite` to `.h`:
   ```bash
   xxd -i model_quant.tflite > model_data.h
   ```

   Place `model_data.h` into `embedded/include/`.

---

## Output Example

After training, the model reached:

- Training Accuracy: ~100%  
- Validation Accuracy: ~83%

Refer to `plots/` for training curves.

---

## Notes

- Use TensorFlow 2.5 for compatibility with TensorFlow Lite Micro.
- Representative dataset for quantization must match the input dimensions (224×224 RGB).
- Do not include batch normalization or dropout in final layers if converting to Microcontroller.

---

## Edits to Make

- Update `train.py` image paths if your dataset is not in `data/shoe/` and `data/non_shoe/`.
- Replace the name in this README if submitting under a different author.
- Use `model_data.h` generated here inside the Arduino project (`embedded/include/`).

---

## Author

Developed by **Sai Venkata Hraswanth Jangam**  
MS Computer Engineering @ UNC Charlotte  
Spring 2025 — ECGR 4127/5127: ML for IoT
