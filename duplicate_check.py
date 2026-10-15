import os
import hashlib
from collections import defaultdict

# ==========================================
# EXACT DUPLICATE IMAGE CHECK
# ==========================================

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Dictionary:
# hash -> list of file paths
image_hashes = defaultdict(list)

total_images = 0

print("=" * 70)
print("        ALZHEIMER DATASET - DUPLICATE CHECK")
print("=" * 70)

# Find class folders
folders = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

print("\nFolders found:")

for folder in folders:
    print(" -", folder)

print("\nScanning images...")
print("-" * 70)

# ==========================================
# CALCULATE SHA-256 HASH FOR EACH IMAGE
# ==========================================

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    print(f"\nProcessing: {folder}")

    folder_count = 0

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(valid_extensions):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:

            sha256 = hashlib.sha256()

            with open(file_path, "rb") as file:

                while True:

                    data = file.read(1024 * 1024)

                    if not data:
                        break

                    sha256.update(data)

            image_hash = sha256.hexdigest()

            image_hashes[image_hash].append(file_path)

            total_images += 1
            folder_count += 1

            # Progress
            if folder_count % 5000 == 0:
                print(
                    f"  {folder}: "
                    f"{folder_count} images checked..."
                )

        except Exception as error:

            print(
                f"ERROR reading: {file_path}"
            )

            print(error)

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

# ==========================================
# FIND DUPLICATES
# ==========================================

duplicate_groups = []

for image_hash, file_list in image_hashes.items():

    if len(file_list) > 1:

        duplicate_groups.append(file_list)

# ==========================================
# RESULTS
# ==========================================

print(f"\nTotal images scanned       : {total_images}")

print(
    f"Unique image hashes       : "
    f"{len(image_hashes)}"
)

print(
    f"Duplicate groups found    : "
    f"{len(duplicate_groups)}"
)

duplicate_image_count = sum(
    len(group)
    for group in duplicate_groups
)

print(
    f"Images involved in duplicates: "
    f"{duplicate_image_count}"
)

# ==========================================
# DISPLAY DUPLICATE GROUPS
# ==========================================

if duplicate_groups:

    print("\nDUPLICATE GROUPS:")
    print("-" * 70)

    for index, group in enumerate(
        duplicate_groups[:20],
        start=1
    ):

        print(f"\nDuplicate Group {index}:")

        for file_path in group:

            print("  ", file_path)

    if len(duplicate_groups) > 20:

        print(
            f"\nOnly first 20 groups displayed."
        )

else:

    print("\nNo exact duplicate images found.")

print("\n" + "=" * 70)
print("DUPLICATE CHECK COMPLETED")
print("=" * 70)