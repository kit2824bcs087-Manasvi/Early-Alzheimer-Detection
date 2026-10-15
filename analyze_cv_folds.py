import os
import re
from collections import defaultdict


# ============================================================
# SUBJECT-LEVEL CROSS-VALIDATION FEASIBILITY CHECK
# ============================================================

DATASET_PATH = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

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
# COLLECT SUBJECTS
# ============================================================

subjects_by_class = defaultdict(set)

images_by_subject = defaultdict(int)


for class_name in CLASS_NAMES:

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    print(
        f"Processing: {class_name}"
    )

    for filename in os.listdir(folder):

        if not filename.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue

        match = SUBJECT_PATTERN.search(
            filename
        )

        if not match:
            continue

        subject_id = match.group(1)

        subjects_by_class[
            class_name
        ].add(
            subject_id
        )

        images_by_subject[
            (class_name, subject_id)
        ] += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT-LEVEL CROSS-VALIDATION FEASIBILITY")
print("=" * 70)


for class_name in CLASS_NAMES:

    subjects = sorted(
        subjects_by_class[class_name]
    )

    print(
        f"\n{class_name}"
    )

    print(
        f"Subjects: {len(subjects)}"
    )


# ============================================================
# FIND MINIMUM CLASS SIZE
# ============================================================

minimum_subject_count = min(

    len(subjects_by_class[class_name])

    for class_name in CLASS_NAMES

)


print("\n" + "=" * 70)

print(
    "Minimum subjects in any class:",
    minimum_subject_count
)

print("=" * 70)


# ============================================================
# K-FOLD CHECK
# ============================================================

print("\nPossible subject-level K-fold configurations:")

for k in range(2, 11):

    possible = (

        minimum_subject_count >= k

    )

    if possible:

        print(
            f"  {k}-fold : POSSIBLE"
        )

    else:

        print(
            f"  {k}-fold : NOT POSSIBLE"
        )


# ============================================================
# MODERATE DEMENTIA DETAILS
# ============================================================

print("\n" + "=" * 70)
print("MODERATE DEMENTIA SUBJECTS")
print("=" * 70)


moderate_subjects = sorted(
    subjects_by_class[
        "Moderate Dementia"
    ]
)


for subject_id in moderate_subjects:

    image_count = images_by_subject[
        (
            "Moderate Dementia",
            subject_id
        )
    ]

    print(
        f"{subject_id}: "
        f"{image_count} images"
    )


# ============================================================
# FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)


if minimum_subject_count >= 5:

    print(
        "At least 5-fold subject-level "
        "cross-validation is feasible."
    )

elif minimum_subject_count >= 3:

    print(
        "3-fold subject-level "
        "cross-validation is feasible."
    )

elif minimum_subject_count >= 2:

    print(
        "Only 2-fold subject-level "
        "cross-validation is feasible."
    )

else:

    print(
        "Subject-level cross-validation "
        "is not feasible."
    )

print("=" * 70)