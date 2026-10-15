import os
from PIL import Image

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

total_images = 0
valid_images = 0
corrupted_images = 0

corrupted_files = []

print("=" * 65)
print("        ALZHEIMER DATASET - IMAGE QUALITY CHECK")
print("=" * 65)

if not os.path.exists(dataset_path):
    print("ERROR: Dataset folder not found!")
    exit()

folders = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("\nFolders found:")
for folder in folders:
    print(" -", folder)

print("\nScanning images...")
print("-" * 65)

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    folder_total = 0
    folder_valid = 0
    folder_corrupted = 0

    print(f"\nStarting: {folder}")

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(valid_extensions):
            continue

        total_images += 1
        folder_total += 1

        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                img.verify()

            valid_images += 1
            folder_valid += 1

        except Exception:
            corrupted_images += 1
            folder_corrupted += 1
            corrupted_files.append(file_path)

        # Show progress every 500 images
        if folder_total % 500 == 0:
            print(
                f"  {folder}: {folder_total} images checked..."
            )

    print(
        f"Completed: {folder} | "
        f"Total: {folder_total} | "
        f"Valid: {folder_valid} | "
        f"Corrupted: {folder_corrupted}"
    )

print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)

print(f"Total images scanned : {total_images}")
print(f"Valid images         : {valid_images}")
print(f"Corrupted images     : {corrupted_images}")

if corrupted_images > 0:

    print("\nCorrupted files:")
    print("-" * 65)

    for file_path in corrupted_files:
        print(file_path)

else:
    print("\nNo corrupted images found.")

print("=" * 65)