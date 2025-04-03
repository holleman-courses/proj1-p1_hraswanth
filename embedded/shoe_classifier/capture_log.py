import serial
import os

port = "COM5"
baudrate = 115200
save_dir = "logs"
os.makedirs(save_dir, exist_ok=True)

ser = serial.Serial(port, baudrate, timeout=5)
image_count = 1
collecting = False
buffer = []

print("Listening for image data...")

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        if "START_IMAGE" in line:
            buffer = ["START_IMAGE"]
            collecting = True
            print(f"Image {image_count}: capture started")
        elif "END_IMAGE" in line:
            buffer.append("END_IMAGE")
            filename = f"image_{image_count:03d}.txt"
            with open(os.path.join(save_dir, filename), "w") as f:
                f.write("\n".join(buffer))
            print(f"Image saved as {filename}")
            image_count += 1
            collecting = False
        elif collecting:
            buffer.append(line)

    except KeyboardInterrupt:
        print("Stopped by user.")
        break