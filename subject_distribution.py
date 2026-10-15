import os
import re
from collections import defaultdict

# ==========================================
# SUBJECT-WISE IMAGE DISTRIBUTION ANALYSIS
# ==========================================

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Extract subject ID from filename
# Example:
# OAS1_0028_MR1_mpr-1_100.jpg
# Subject ID = OAS1_0028

subject_pattern = re.compile(r"(OAS1_\d+)")

# Store image count for each subject
subject_image_counts = defaultdict(lambda: defaultdict(int))

print("=" * 75)
print("       SUBJECT-WISE IMAGE DISTRIBUTION ANALYSIS")
print("=" * 75)

# Find class folders
folders = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

# Scan dataset
for class_name in folders:

    folder_path = os.path.join(dataset_path, class_name)

    print("\nProcessing:", class_name)

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(valid_extensions):
            continue

        match = subject_pattern.search(file_name)

        if match:

            subject_id = match.group(1)

            subject_image_counts[class_name][subject_id] += 1


# ==========================================
# DISPLAY SUBJECT DISTRIBUTION
# ==========================================

print("\n" + "=" * 75)
print("SUBJECT IMAGE COUNTS")
print("=" * 75)

for class_name in folders:

    print("\n" + "-" * 75)
    print("CLASS:", class_name)
    print("-" * 75)

    subjects = subject_image_counts[class_name]

    # Sort subjects by ID
    sorted_subjects = sorted(subjects.items())

    for subject_id, image_count in sorted_subjects:

        print(
            f"{subject_id:15s} : "
            f"{image_count:4d} images"
        )


# ==========================================
# SUMMARY
# ==========================================

print("\n" + "=" * 75)
print("CLASS-WISE SUMMARY")
print("=" * 75)

for class_name in folders:

    subjects = subject_image_counts[class_name]

    if len(subjects) == 0:
        continue

    image_counts = list(subjects.values())

    total_images = sum(image_counts)
    number_of_subjects = len(image_counts)

    minimum_images = min(image_counts)
    maximum_images = max(image_counts)

    average_images = total_images / number_of_subjects

    print(f"\n{class_name}")
    print(f"  Subjects       : {number_of_subjects}")
    print(f"  Total images   : {total_images}")
    print(f"  Minimum images : {minimum_images}")
    print(f"  Maximum images : {maximum_images}")
    print(f"  Average images : {average_images:.2f}")

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)