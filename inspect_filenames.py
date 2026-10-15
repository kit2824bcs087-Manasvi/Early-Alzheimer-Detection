import os

# ==========================================
# INSPECT DATASET FILENAMES
# ==========================================

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

print("=" * 70)
print("        ALZHEIMER DATASET - FILENAME ANALYSIS")
print("=" * 70)

# Find class folders
folders = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    print("\n" + "-" * 70)
    print("CLASS:", folder)
    print("-" * 70)

    image_files = []

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_files.append(file_name)

    print("Total images:", len(image_files))

    print("\nFirst 10 filenames:")

    for file_name in image_files[:10]:
        print(file_name)

print("\n" + "=" * 70)
print("FILENAME ANALYSIS COMPLETED")
print("=" * 70)
