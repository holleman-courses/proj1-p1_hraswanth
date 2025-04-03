import os
import numpy as np
from PIL import Image

# === Configuration ===
WIDTH, HEIGHT, CHANNELS = 320, 240, 3
input_dir = "logs"
output_dir = "converted_images"
os.makedirs(output_dir, exist_ok=True)

def rgb565_to_rgb888(byte1, byte2):
    value = (byte1 << 8) | byte2
    r = ((value >> 11) & 0x1F) * 255 // 31
    g = ((value >> 5) & 0x3F) * 255 // 63
    b = (value & 0x1F) * 255 // 31
    return [r, g, b]

def convert_file_to_image(filepath, index):
    with open(filepath, "r") as f:
        content = f.read()

    try:
        hex_data = content.split("START_IMAGE")[1].split("END_IMAGE")[0].strip().split()
    except IndexError:
        print(f"Skipping {filepath} — markers not found")
        return False

    byte_data = []
    for h in hex_data:
        try:
            byte_data.append(int(h, 16))
        except ValueError:
            continue

    if len(byte_data) < WIDTH * HEIGHT * 2:
        print(f"Skipping {filepath} — not enough data ({len(byte_data)} bytes)")
        return False

    rgb888 = []
    for i in range(0, WIDTH * HEIGHT * 2, 2):
        rgb888.extend(rgb565_to_rgb888(byte_data[i], byte_data[i + 1]))

    img_array = np.array(rgb888, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
    img = Image.fromarray(img_array)
    output_path = os.path.join(output_dir, f"img_{index:03d}.png")
    img.save(output_path)
    print(f"Saved: {output_path}")
    return True

# === Process all files ===
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])
count = 0

for i, file in enumerate(image_files, start=1):
    filepath = os.path.join(input_dir, file)
    if convert_file_to_image(filepath, i):
        count += 1

print(f"\nDone! {count} image(s) saved to '{output_dir}' folder.")