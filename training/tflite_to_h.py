import sys

# === Input/Output paths ===
input_tflite_path = "model_float16.tflite"  # change this if needed
output_header_path = "model_data.h"
variable_name = "g_model"  # this name will be used as your model array

# === Read .tflite model as bytes ===
with open(input_tflite_path, "rb") as f:
    model_bytes = f.read()

# === Write to .h file ===
with open(output_header_path, "w") as f:
    f.write(f'#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n')
    f.write(f'alignas(16) const unsigned char {variable_name}[] = {{\n    ')

    for i, byte in enumerate(model_bytes):
        f.write(f'{byte}, ')
        if (i + 1) % 12 == 0:
            f.write('\n    ')

    f.write(f'\n}};\n\n')
    f.write(f'const int {variable_name}_len = {len(model_bytes)};\n')
    f.write(f'\n#endif  // MODEL_DATA_H\n')

print(f"✅ Header file saved to {output_header_path}")