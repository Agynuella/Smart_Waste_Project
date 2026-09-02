import os
import urllib.request
import zipfile
import shutil
import random

# 1. CONFIGURATION
URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
ZIP_NAME = "dataset-resized.zip"
EXTRACT_DIR = "dataset-resized"
FINAL_DIR = "trashnet_dataset"

def main():
    # 2. DOWNLOAD & EXTRACT
    print("📥 Downloading TrashNet dataset (~40MB)... This might take a minute.")
    urllib.request.urlretrieve(URL, ZIP_NAME)
    print("📦 Download complete. Extracting files...")
    
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(".")

    # 3. CREATE DIRECTORY STRUCTURE (Train/Val)
    print("🗂️ Organizing images into Training (80%) and Validation (20%) sets...")
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    for phase in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(FINAL_DIR, phase, cls), exist_ok=True)

    # 4. SHUFFLE AND SPLIT IMAGES
    for cls in classes:
        src_dir = os.path.join(EXTRACT_DIR, cls)
        # Ensure we only read actual image files
        images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
        random.shuffle(images) # Shuffle so the AI doesn't memorize the order
        
        split_index = int(0.8 * len(images)) # 80% for training
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # Copy to train folder
        for img in train_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(FINAL_DIR, 'train', cls, img))
            
        # Copy to val folder
        for img in val_images:
            shutil.copy(os.path.join(src_dir, img), os.path.join(FINAL_DIR, 'val', cls, img))

    # 5. CLEANUP
    print("🧹 Cleaning up temporary files...")
    os.remove(ZIP_NAME)
    shutil.rmtree(EXTRACT_DIR)

    print(f"✅ Success! 2,500+ images are now perfectly organized inside the '{FINAL_DIR}' folder.")

if __name__ == '__main__':
    main()