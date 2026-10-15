import os
import re
import random
import csv
from collections import defaultdict

# ============================================================
# ALZHEIMER DATASET - SUBJECT LEVEL DATASET SPLIT
# ============================================================

DATASET_PATH = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

OUTPUT_FILE = "dataset_split.csv"

SEED = 42

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

# Subject ID pattern
# Example:
# OAS1_0028_MR1_mpr-1_100.jpg
#
# Subject ID:
# OAS1_0028

SUBJECT_PATTERN = re.compile(r"(OAS1_\d+)")

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

print("=" * 75)
print("       ALZHEIMER DATASET - SUBJECT LEVEL SPLIT")
print("=" * 75)

print("\nRandom seed:", SEED)
print("Dataset:", DATASET_PATH)

# ============================================================
# CHECK DATASET PATH
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset folder not found!")
    print(DATASET_PATH)
    exit()

# ============================================================
# FIND CLASS FOLDERS
# ============================================================

class_folders = [
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
]

print("\nClasses found:")

for class_name in class_folders:
    print(" -", class_name)

# ============================================================
# COLLECT IMAGES BY SUBJECT
# ============================================================

subjects_by_class = defaultdict(
    lambda: defaultdict(list)
)

total_images = 0

print("\nCollecting images by subject...")
print("-" * 75)

for class_name in class_folders:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for file_name in os.listdir(class_path):

        if not file_name.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        match = SUBJECT_PATTERN.search(file_name)

        if not match:

            print(
                "WARNING: Subject ID not found:",
                file_name
            )

            continue

        subject_id = match.group(1)

        file_path = os.path.join(
            class_path,
            file_name
        )

        subjects_by_class[
            class_name
        ][subject_id].append(file_path)

        total_images += 1

# ============================================================
# SPLIT SUBJECTS
# ============================================================

split_records = []

subject_split_summary = []

print("\nCreating subject-level splits...")
print("-" * 75)

for class_name in class_folders:

    subjects = list(
        subjects_by_class[class_name].keys()
    )

    random.shuffle(subjects)

    number_of_subjects = len(subjects)

    # --------------------------------------------------------
    # SPECIAL CASE: VERY SMALL CLASS
    # --------------------------------------------------------

    if number_of_subjects == 2:

        # One subject for training
        # One subject for testing
        #
        # No validation subject because only
        # two independent subjects exist.

        train_subjects = subjects[:1]

        validation_subjects = []

        test_subjects = subjects[1:]

    else:

        # ----------------------------------------------------
        # Approximately 70 / 15 / 15 SUBJECT SPLIT
        # ----------------------------------------------------

        train_count = int(
            number_of_subjects * 0.70
        )

        validation_count = int(
            number_of_subjects * 0.15
        )

        # Make sure we don't accidentally create
        # an empty training set.

        if train_count < 1:
            train_count = 1

        train_subjects = subjects[
            :train_count
        ]

        validation_subjects = subjects[
            train_count:
            train_count + validation_count
        ]

        test_subjects = subjects[
            train_count + validation_count:
        ]

    # --------------------------------------------------------
    # CREATE IMAGE RECORDS
    # --------------------------------------------------------

    for subject_id in train_subjects:

        for file_path in subjects_by_class[
            class_name
        ][subject_id]:

            split_records.append({
                "image_path": file_path,
                "subject_id": subject_id,
                "class": class_name,
                "split": "train"
            })

    for subject_id in validation_subjects:

        for file_path in subjects_by_class[
            class_name
        ][subject_id]:

            split_records.append({
                "image_path": file_path,
                "subject_id": subject_id,
                "class": class_name,
                "split": "validation"
            })

    for subject_id in test_subjects:

        for file_path in subjects_by_class[
            class_name
        ][subject_id]:

            split_records.append({
                "image_path": file_path,
                "subject_id": subject_id,
                "class": class_name,
                "split": "test"
            })

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    subject_split_summary.append({
        "class": class_name,
        "total_subjects": number_of_subjects,
        "train_subjects": len(train_subjects),
        "validation_subjects": len(
            validation_subjects
        ),
        "test_subjects": len(test_subjects)
    })

# ============================================================
# WRITE CSV FILE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "image_path",
        "subject_id",
        "class",
        "split"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(split_records)

# ============================================================
# DISPLAY SUBJECT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SUBJECT SPLIT SUMMARY")
print("=" * 75)

for summary in subject_split_summary:

    print(
        f"\n{summary['class']}"
    )

    print(
        f"  Total subjects      : "
        f"{summary['total_subjects']}"
    )

    print(
        f"  Train subjects      : "
        f"{summary['train_subjects']}"
    )

    print(
        f"  Validation subjects : "
        f"{summary['validation_subjects']}"
    )

    print(
        f"  Test subjects       : "
        f"{summary['test_subjects']}"
    )

# ============================================================
# IMAGE SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("IMAGE SPLIT SUMMARY")
print("=" * 75)

image_counts = defaultdict(int)

for record in split_records:

    image_counts[
        record["split"]
    ] += 1

for split_name in [
    "train",
    "validation",
    "test"
]:

    print(
        f"{split_name:15s}: "
        f"{image_counts[split_name]}"
    )

print(
    f"{'TOTAL':15s}: "
    f"{sum(image_counts.values())}"
)

# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 75)
print("SUBJECT LEAKAGE CHECK")
print("=" * 75)

subjects_in_split = defaultdict(set)

for record in split_records:

    subjects_in_split[
        record["split"]
    ].add(record["subject_id"])

train_subjects = subjects_in_split["train"]
validation_subjects = subjects_in_split["validation"]
test_subjects = subjects_in_split["test"]

train_validation_overlap = (
    train_subjects &
    validation_subjects
)

train_test_overlap = (
    train_subjects &
    test_subjects
)

validation_test_overlap = (
    validation_subjects &
    test_subjects
)

print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)

if (
    len(train_validation_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(validation_test_overlap) == 0
):

    print(
        "\nPASS: No subject leakage detected."
    )

else:

    print(
        "\nWARNING: Subject leakage detected!"
    )

print("\n" + "=" * 75)
print("SPLIT CREATION COMPLETED")
print("=" * 75)

print(
    "\nSplit file created:"
)

print(
    os.path.abspath(OUTPUT_FILE)
)