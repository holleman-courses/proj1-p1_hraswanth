import os
import requests
from io import BytesIO
from PIL import Image

# User-Agent header to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Create dataset folders
os.makedirs(os.path.join(DATASET_DIR, "shoe"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "non_shoe"), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, "test_images"), exist_ok=True)

def download_images(image_urls, folder, prefix):
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            img = img.resize((28, 28))

            img_path = os.path.join(DATASET_DIR, folder, f"{prefix}{i+1}.jpg")
            img.save(img_path)
            print(f"Saved {img_path}")

        except Exception as e:
            print(f"Error downloading {url}: {e}")

# 5 Training Shoe Images
shoe_urls = [
    "https://images.pexels.com/photos/267202/pexels-photo-267202.jpeg",
    "https://images.pexels.com/photos/267242/pexels-photo-267242.jpeg",
    "https://images.pexels.com/photos/998592/pexels-photo-998592.jpeg",
    "https://images.pexels.com/photos/19090/pexels-photo.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/96/Army-boots.jpg"
]

# 5 Training Non-Shoe Images
non_shoe_urls = [
    "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg",
    "https://upload.wikimedia.org/wikipedia/commons/6/6b/Sunflower.jpg",
    "https://images.pexels.com/photos/206359/pexels-photo-206359.jpeg",
    "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg",
    "https://images.pexels.com/photos/91216/pexels-photo-91216.jpeg"
]

# ✅ 10 New Verified Test Shoe Images
test_shoe_urls = [
    "https://images.pexels.com/photos/276516/pexels-photo-276516.jpeg",
    "https://images.pexels.com/photos/2529148/pexels-photo-2529148.jpeg",
    "https://images.pexels.com/photos/1124466/pexels-photo-1124466.jpeg",
    "https://images.pexels.com/photos/1134176/pexels-photo-1134176.jpeg",
    "https://images.pexels.com/photos/292999/pexels-photo-292999.jpeg",
    "https://images.pexels.com/photos/298863/pexels-photo-298863.jpeg",
    "https://images.pexels.com/photos/4342420/pexels-photo-4342420.jpeg",
    "https://images.pexels.com/photos/189496/pexels-photo-189496.jpeg",
    "https://images.pexels.com/photos/356722/pexels-photo-356722.jpeg",
    "https://images.pexels.com/photos/1598505/pexels-photo-1598505.jpeg"
]

# ✅ 10 New Verified Test Non-Shoe Images
test_non_shoe_urls = [
    "https://images.pexels.com/photos/17390173/pexels-photo-17390173/free-photo-of-a-black-and-white-photo-of-a-car.jpeg",
    "https://images.pexels.com/photos/301496/pexels-photo-301496.jpeg",
    "https://images.pexels.com/photos/206359/pexels-photo-206359.jpeg",
    "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg",
    "https://images.pexels.com/photos/356079/pexels-photo-356079.jpeg",
    "https://images.pexels.com/photos/262508/pexels-photo-262508.jpeg",
    "https://images.pexels.com/photos/2395257/pexels-photo-2395257.jpeg",
    "https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg",
    "https://images.pexels.com/photos/280310/pexels-photo-280310.jpeg",
    "https://images.pexels.com/photos/258293/pexels-photo-258293.jpeg"
]

# Download images
download_images(shoe_urls, "shoe", "Sh")
download_images(non_shoe_urls, "non_shoe", "Ns")
download_images(test_shoe_urls, "test_images", "Ts")
download_images(test_non_shoe_urls, "test_images", "Tns")

print("Dataset preparation complete.")