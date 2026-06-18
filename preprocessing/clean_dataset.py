from PIL import Image
import os

DATASET_PATH = "../data"

removed = 0

for root, _, files in os.walk(DATASET_PATH):
    for file in files:

        path = os.path.join(root, file)

        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img.save(path)

        except Exception:
            print("Removing:", path)
            os.remove(path)
            removed += 1

print("Removed files:", removed)