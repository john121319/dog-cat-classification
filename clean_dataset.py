import os
from PIL import Image

DATASET_PATH = "dogs-vs-cats-classification"

def clean_dataset(root):
    removed = 0

    for folder, _, files in os.walk(root):
        for file in files:
            path = os.path.join(folder, file)

            try:
                # open image
                img = Image.open(path)

                # force convert to RGB (fixes 2-channel issue)
                img = img.convert("RGB")

                # re-save clean image (overwrite)
                img.save(path)

            except Exception as e:
                print("Deleting bad file:", path)
                os.remove(path)
                removed += 1

    print("\nDONE. Removed:", removed, "corrupted files")

clean_dataset(DATASET_PATH)