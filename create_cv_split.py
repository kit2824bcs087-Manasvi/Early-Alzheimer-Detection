import os
import re
import csv
import random
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

OUTPUT_FILE = "cv_split.csv"

SEED = 42

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

SUBJECT_PATTERN = re.compile(
    r"(OAS1_\d+)"
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)


print("=" * 75)
print("       ALZHEIMER DATASET - 2-FOLD SUBJECT CV SPLIT")
print("=" * 75)

print("\nRandom seed:", SEED)
print("Dataset:", DATASET_PATH)


# ============================================================
# COLLECT IMAGES BY SUBJECT
# ============================================================

subjects = defaultdict(
    lambda: defaultdict(list)
)


print("\n" + "=" * 75)
print("COLLECTING IMAGES BY SUBJECT")
print("=" * 75)


for class_name in CLASS_NAMES:

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(folder_path):

        print(
            "ERROR: Folder not found:",
            folder_path
        )

        continue


    print(
        "\nProcessing:",
        class_name
    )


    for file_name in os.listdir(
        folder_path
    ):

        if not file_name.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue


        match = SUBJECT_PATTERN.search(
            file_name
        )

        if not match:
            continue


        subject_id = match.group(1)


        image_path = os.path.join(
            folder_path,
            file_name
        )


        subjects[class_name][
            subject_id
        ].append(
            image_path
        )


# ============================================================
# CREATE 2 FOLDS
# ============================================================

print("\n" + "=" * 75)
print("CREATING 2-FOLD SUBJECT SPLITS")
print("=" * 75)


fold_subjects = {
    1: defaultdict(list),
    2: defaultdict(list)
}


for class_name in CLASS_NAMES:

    class_subjects = list(
        subjects[class_name].keys()
    )


    random.shuffle(
        class_subjects
    )


    midpoint = len(class_subjects) // 2


    fold1_subjects = class_subjects[
        :midpoint
    ]

    fold2_subjects = class_subjects[
        midpoint:
    ]


    fold_subjects[1][
        class_name
    ] = fold1_subjects


    fold_subjects[2][
        class_name
    ] = fold2_subjects


# ============================================================
# DISPLAY SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("FOLD SUBJECT DISTRIBUTION")
print("=" * 75)


for fold in [1, 2]:

    print(
        f"\nFOLD {fold}"
    )

    print("-" * 75)


    total_subjects = 0


    for class_name in CLASS_NAMES:

        count = len(
            fold_subjects[
                fold
            ][
                class_name
            ]
        )


        total_subjects += count


        print(
            f"{class_name:25s}: "
            f"{count:3d} subjects"
        )


    print(
        f"TOTAL: {total_subjects} subjects"
    )


# ============================================================
# MODERATE DEMENTIA CHECK
# ============================================================

print("\n" + "=" * 75)
print("MODERATE DEMENTIA CHECK")
print("=" * 75)


for fold in [1, 2]:

    moderate_subjects = fold_subjects[
        fold
    ][
        "Moderate Dementia"
    ]


    print(
        f"\nFold {fold}:"
    )


    for subject_id in moderate_subjects:

        image_count = len(
            subjects[
                "Moderate Dementia"
            ][
                subject_id
            ]
        )


        print(
            f"  {subject_id}: "
            f"{image_count} images"
        )


# ============================================================
# WRITE CSV
# ============================================================

print("\n" + "=" * 75)
print("CREATING CSV")
print("=" * 75)


rows = []


for fold in [1, 2]:

    validation_subjects = set()


    for class_name in CLASS_NAMES:

        validation_subjects.update(
            fold_subjects[
                fold
            ][
                class_name
            ]
        )


    for class_name in CLASS_NAMES:

        for subject_id, image_paths in subjects[
            class_name
        ].items():


            if subject_id in validation_subjects:

                split = "validation"

            else:

                split = "train"


            for image_path in image_paths:

                rows.append({

                    "fold": fold,

                    "split": split,

                    "subject_id": subject_id,

                    "class": class_name,

                    "image_path": image_path

                })


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "fold",
            "split",
            "subject_id",
            "class",
            "image_path"
        ]

    )


    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# FINAL CHECK
# ============================================================

print(
    "\nCSV created:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 75)
print("FINAL CHECK")
print("=" * 75)


for fold in [1, 2]:

    train_subjects = set()
    validation_subjects = set()


    for row in rows:

        if int(row["fold"]) != fold:
            continue


        if row["split"] == "train":

            train_subjects.add(
                row["subject_id"]
            )

        else:

            validation_subjects.add(
                row["subject_id"]
            )


    overlap = (
        train_subjects
        &
        validation_subjects
    )


    print(
        f"\nFold {fold}"
    )

    print(
        "Train subjects:",
        len(train_subjects)
    )

    print(
        "Validation subjects:",
        len(validation_subjects)
    )

    print(
        "Subject overlap:",
        len(overlap)
    )


    if len(overlap) == 0:

        print(
            "PASS: No subject leakage"
        )

    else:

        print(
            "FAIL: Subject leakage detected!"
        )


print("\n" + "=" * 75)
print("2-FOLD SUBJECT CV SPLIT COMPLETED")
print("=" * 75)