import os
import random
import shutil
from icrawler.builtin import GoogleImageCrawler

# -------------------------
# HARDCODED CAR MODELS
# -------------------------
car_models = {
    "BMW": [
        "3 Series", "5 Series", "7 Series", "X3", "X5",
        "X7", "M3", "M5", "i3", "i8"
    ],
    "Mercedes": [
        "A-Class", "C-Class", "E-Class", "S-Class", "G-Class",
        "GLC", "GLE", "GLS", "AMG GT", "EQC"
    ],
    "Volkswagen": [
        "Golf", "Passat", "Polo", "Arteon", "Jetta",
        "Tiguan", "Touareg", "ID.3", "ID.4", "Beetle"
    ],
    "Audi": [
        "A3", "A4", "A6", "A8", "Q3",
        "Q5", "Q7", "Q8", "TT", "R8"
    ],
    "Porsche": [
        "911", "718 Cayman", "718 Boxster", "Panamera", "Macan",
        "Cayenne", "Taycan", "918 Spyder", "Carrera GT", "356"
    ]
}

# -------------------------
# IMAGE DOWNLOAD FUNCTION
# -------------------------
def download_images(brand, model, num_images=100, base_dir="Cars/train"):
    """Download images for a given brand + model."""
    save_dir = os.path.join(base_dir, brand, model.replace("/", "-"))
    os.makedirs(save_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=f"{brand} {model} car", max_num=num_images)


# -------------------------
# TRAIN/TEST SPLIT FUNCTION
# -------------------------
def split_train_test(base_dir="Cars", split_ratio=0.8):
    """Move 20% of images to test set."""
    for brand in os.listdir(os.path.join(base_dir, "train")):
        brand_dir = os.path.join(base_dir, "train", brand)
        for model in os.listdir(brand_dir):
            model_dir = os.path.join(brand_dir, model)
            images = os.listdir(model_dir)
            if not images:
                continue
            random.shuffle(images)
            
            split_point = int(len(images) * split_ratio)
            test_images = images[split_point:]
            
            test_dir = os.path.join(base_dir, "test", brand, model)
            os.makedirs(test_dir, exist_ok=True)
            
            for img in test_images:
                src = os.path.join(model_dir, img)
                dst = os.path.join(test_dir, img)
                shutil.move(src, dst)


# -------------------------
# MAIN PIPELINE
# -------------------------
if __name__ == "__main__":
    # 1. Download images
    for brand, models in car_models.items():
        for model in models:
            print(f"⬇️ Downloading {brand} {model} ...")
            try:
                download_images(brand, model, num_images=100)  # change number of images here
            except Exception as e:
                print(f"❌ Failed {brand} {model}: {e}")
    
    # 2. Split into train/test
    print("📂 Splitting into train/test sets...")
    split_train_test()
    print("✅ Dataset ready in ./Cars/")
