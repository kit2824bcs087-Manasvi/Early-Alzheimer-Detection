import csv
from collections import defaultdict

# ==========================================
# DATASET SPLIT ANALYSIS
# ==========================================

split_file = "dataset_split.csv"

# Store image counts
image_counts = defaultdict(
    lambda: defaultdict(int)
)

# Store unique subjects
subjects = defaultdict(
    lambda: defaultdict(set)
)

total_images = 0

print("=" * 75)
print("          ALZHEIMER DATASET - SPLIT ANALYSIS")
print("=" * 75)

# ==========================================
# READ CSV
# ==========================================

try:

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            split = row["split"]
            class_name = row["class"]
            subject_id = row["subject_id"]

            image_counts[split][class_name] += 1

            subjects[split][class_name].add(
                subject_id
            )

            total_images += 1

except FileNotFoundError:

    print("\nERROR: dataset_split.csv not found!")
    print(
        "Make sure this script is in the same "
        "folder as dataset_split.csv."
    )
    exit()


# ==========================================
# CLASS ORDER
# ==========================================

class_order = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

split_order = [
    "train",
    "validation",
    "test"
]


# ==========================================
# IMAGE DISTRIBUTION
# ==========================================

print("\n" + "=" * 75)
print("IMAGE DISTRIBUTION")
print("=" * 75)

for split in split_order:

    print("\n" + "-" * 75)
    print(split.upper())
    print("-" * 75)

    split_total = 0

    for class_name in class_order:

        count = image_counts[
            split
        ][class_name]

        print(
            f"{class_name:25s}: {count:6d}"
        )

        split_total += count

    print(
        f"{'TOTAL':25s}: {split_total:6d}"
    )


# ==========================================
# SUBJECT DISTRIBUTION
# ==========================================

print("\n" + "=" * 75)
print("SUBJECT DISTRIBUTION")
print("=" * 75)

for split in split_order:

    print("\n" + "-" * 75)
    print(split.upper())
    print("-" * 75)

    split_subject_total = 0

    for class_name in class_order:

        count = len(
            subjects[split][class_name]
        )

        print(
            f"{class_name:25s}: {count:6d} subjects"
        )

        split_subject_total += count

    print(
        f"{'TOTAL':25s}: "
        f"{split_subject_total:6d} subjects"
    )


# ==========================================
# CLASS PERCENTAGES
# ==========================================

print("\n" + "=" * 75)
print("CLASS PERCENTAGE WITHIN EACH SPLIT")
print("=" * 75)

for split in split_order:

    split_total = sum(
        image_counts[split].values()
    )

    print("\n" + split.upper())

    for class_name in class_order:

        count = image_counts[
            split
        ][class_name]

        if split_total > 0:

            percentage = (
                count / split_total
            ) * 100

        else:

            percentage = 0

        print(
            f"{class_name:25s}: "
            f"{percentage:6.2f}%"
        )


# ==========================================
# FINAL TOTAL
# ==========================================

print("\n" + "=" * 75)
print("FINAL CHECK")
print("=" * 75)

train_total = sum(
    image_counts["train"].values()
)

validation_total = sum(
    image_counts["validation"].values()
)

test_total = sum(
    image_counts["test"].values()
)

print(
    f"Train images       : {train_total}"
)

print(
    f"Validation images  : {validation_total}"
)

print(
    f"Test images        : {test_total}"
)

print(
    f"Total images       : "
    f"{train_total + validation_total + test_total}"
)

print(
    f"Expected images    : 86437"
)

if (
    train_total
    + validation_total
    + test_total
) == 86437:

    print("\nPASS: All 86,437 images are included.")

else:

    print(
        "\nWARNING: Image count mismatch!"
    )

print("=" * 75)
print("SPLIT ANALYSIS COMPLETED")
print("=" * 75)