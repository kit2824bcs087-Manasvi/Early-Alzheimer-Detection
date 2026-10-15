import os
import re
from collections import defaultdict

# ==========================================
# ALZHEIMER DATASET - SUBJECT ANALYSIS
# ==========================================

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

subject_pattern = re.compile(r"(OAS1_\d+)")

subjects_by_class = defaultdict(set)

total_images = 0

print("=" * 70)
print("        ALZHEIMER DATASET - SUBJECT ANALYSIS")
print("=" * 70)

folders = [
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
]

for folder in folders:

    folder_path = os.path.join(dataset_path, folder)

    print("\nProcessing:", folder)

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(valid_extensions):
            continue

        total_images += 1

        match = subject_pattern.search(file_name)

        if match:
            subject_id = match.group(1)
            subjects_by_class[folder].add(subject_id)

print("\n" + "=" * 70)
print("SUBJECT SUMMARY")
print("=" * 70)

all_subjects = set()

for class_name in folders:

    subject_count = len(subjects_by_class[class_name])

    all_subjects.update(subjects_by_class[class_name])

    print(
        f"{class_name:25s}: "
        f"{subject_count} unique subjects"
    )

print("-" * 70)

print(f"Total images         : {total_images}")
print(f"Total unique subjects: {len(all_subjects)}")

print("=" * 70)

print("\nCHECKING SUBJECT OVERLAP BETWEEN CLASSES...")
print("-" * 70)

subject_classes = defaultdict(set)

for class_name in subjects_by_class:

    for subject_id in subjects_by_class[class_name]:

        subject_classes[subject_id].add(class_name)

overlapping_subjects = {
    subject_id: classes
    for subject_id, classes in subject_classes.items()
    if len(classes) > 1
}

if overlapping_subjects:

    print("WARNING: Some subjects appear in multiple classes!")

    for subject_id, classes in list(
        overlapping_subjects.items()
    )[:20]:

        print(
            subject_id,
            "->",
            ", ".join(classes)
        )

    print(
        f"\nTotal overlapping subjects: "
        f"{len(overlapping_subjects)}"
    )

else:

    print("No subject was found in multiple classes.")

print("=" * 70)