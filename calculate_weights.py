import csv
from collections import Counter

# ==========================================
# ALZHEIMER DATASET - CLASS WEIGHT CALCULATION
# ==========================================

split_file = "dataset_split.csv"

print("=" * 70)
print("       ALZHEIMER DATASET - CLASS WEIGHTS")
print("=" * 70)

# ------------------------------------------------
# Read training labels only
# ------------------------------------------------

train_labels = []

try:

    with open(
        split_file,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["split"] == "train":
                train_labels.append(row["class"])

except FileNotFoundError:

    print("\nERROR: dataset_split.csv not found!")
    exit()


# ------------------------------------------------
# Count training images
# ------------------------------------------------

class_counts = Counter(train_labels)

classes = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

total_train = len(train_labels)

number_of_classes = len(classes)

print("\nTraining image counts:")
print("-" * 70)

for class_name in classes:

    count = class_counts[class_name]

    print(
        f"{class_name:25s}: {count}"
    )


# ------------------------------------------------
# Calculate balanced class weights
#
# weight = total_samples /
#          (number_of_classes × class_samples)
# ------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATED CLASS WEIGHTS")
print("=" * 70)

weights = {}

for class_name in classes:

    count = class_counts[class_name]

    weight = total_train / (
        number_of_classes * count
    )

    weights[class_name] = weight

    print(
        f"{class_name:25s}: "
        f"{weight:.6f}"
    )


# ------------------------------------------------
# Normalized weights
# ------------------------------------------------

print("\n" + "=" * 70)
print("NORMALIZED CLASS WEIGHTS")
print("=" * 70)

average_weight = sum(
    weights.values()
) / number_of_classes

for class_name in classes:

    normalized_weight = (
        weights[class_name] /
        average_weight
    )

    print(
        f"{class_name:25s}: "
        f"{normalized_weight:.6f}"
    )


# ------------------------------------------------
# PyTorch order
# ------------------------------------------------

print("\n" + "=" * 70)
print("PYTORCH CLASS WEIGHTS")
print("=" * 70)

print("\nClass order:")

for index, class_name in enumerate(classes):

    print(
        f"{index}: {class_name}"
    )

print("\nWeights:")

pytorch_weights = [
    weights[class_name]
    for class_name in classes
]

print(pytorch_weights)

print("\n" + "=" * 70)
print("CLASS WEIGHT CALCULATION COMPLETED")
print("=" * 70)