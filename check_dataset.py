import os

# Dataset location
dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

total_images = 0

print("=" * 60)
print("          ALZHEIMER DATASET ANALYSIS")
print("=" * 60)

# Check dataset folder
if not os.path.exists(dataset_path):
    print("ERROR: Dataset folder not found!")
    print(dataset_path)
    exit()

# Automatically find all folders
folders = [
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("\nFolders found:")

for folder in folders:
    print(" -", folder)

print("\nImage count:")
print("-" * 60)

# Count images
for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    count = 0

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(valid_extensions):
            count += 1

    print(f"{folder:25s}: {count}")

    total_images += count

print("-" * 60)
print(f"{'TOTAL IMAGES':25s}: {total_images}")
print("=" * 60)