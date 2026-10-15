import os
import re
from collections import defaultdict

# ============================================================
# SUBJECT-WISE IMAGE DISTRIBUTION ANALYSIS
# ============================================================

dataset_path = r"C:\Users\ROHITH-PC\Downloads\alzhemeir-ai-project\Data"

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

# Example filename:
# OAS1_0028_MR1_mpr-1_100.jpg
#
# Subject ID:
# OAS1_0028

subject_pattern = re.compile(
    r"(OAS1_\d+)"
)


# ============================================================
# STORE SUBJECT INFORMATION
# ============================================================

subject_image_counts = defaultdict(
    lambda: defaultdict(int)
)


print("=" * 75)
print("       SUBJECT-WISE IMAGE DISTRIBUTION ANALYSIS")
print("=" * 75)


print("\nDataset:")
print(dataset_path)


# ============================================================
# CHECK DATASET PATH
# ============================================================

if not os.path.exists(dataset_path):

    print("\nERROR:")
    print("Dataset folder not found!")

    print(
        "\nCheck this path:"
    )

    print(dataset_path)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# FIND CLASS FOLDERS
# ============================================================

class_names = [

    "Mild Dementia",

    "Moderate Dementia",

    "Non Demented",

    "Very mild Dementia"

]


print("\nFolders found:")


for class_name in class_names:

    folder_path = os.path.join(

        dataset_path,

        class_name

    )

    if os.path.isdir(folder_path):

        print(
            " -",
            class_name
        )

    else:

        print(
            " -",
            class_name,
            "NOT FOUND"
        )


# ============================================================
# SCAN IMAGES
# ============================================================

print("\n" + "=" * 75)
print("SCANNING SUBJECTS")
print("=" * 75)


total_images = 0


for class_name in class_names:

    folder_path = os.path.join(

        dataset_path,

        class_name

    )


    if not os.path.isdir(folder_path):

        continue


    print(
        "\nProcessing:",
        class_name
    )


    image_counter = 0


    for file_name in os.listdir(
        folder_path
    ):


        if not file_name.lower().endswith(
            valid_extensions
        ):

            continue


        match = subject_pattern.search(
            file_name
        )


        if not match:

            print(
                "WARNING: Subject ID not found:",
                file_name
            )

            continue


        subject_id = match.group(1)


        subject_image_counts[
            class_name
        ][
            subject_id
        ] += 1


        image_counter += 1

        total_images += 1


    print(
        "Images processed:",
        image_counter
    )


# ============================================================
# SUBJECT DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("SUBJECT IMAGE COUNTS")
print("=" * 75)


for class_name in class_names:


    print("\n" + "-" * 75)

    print(
        "CLASS:",
        class_name
    )

    print("-" * 75)


    subjects = subject_image_counts[
        class_name
    ]


    sorted_subjects = sorted(
        subjects.items()
    )


    for subject_id, image_count in sorted_subjects:

        print(

            f"{subject_id:15s} : "
            f"{image_count:4d} images"

        )


# ============================================================
# CLASS-WISE SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("CLASS-WISE SUMMARY")
print("=" * 75)


for class_name in class_names:


    subjects = subject_image_counts[
        class_name
    ]


    if len(subjects) == 0:

        continue


    image_counts = list(
        subjects.values()
    )


    total_class_images = sum(
        image_counts
    )


    number_of_subjects = len(
        image_counts
    )


    minimum_images = min(
        image_counts
    )


    maximum_images = max(
        image_counts
    )


    average_images = (

        total_class_images
        /
        number_of_subjects

    )


    print(
        f"\n{class_name}"
    )


    print(
        f"  Subjects       : "
        f"{number_of_subjects}"
    )


    print(
        f"  Total images   : "
        f"{total_class_images}"
    )


    print(
        f"  Minimum images : "
        f"{minimum_images}"
    )


    print(
        f"  Maximum images : "
        f"{maximum_images}"
    )


    print(
        f"  Average images : "
        f"{average_images:.2f}"
    )


# ============================================================
# SPECIAL FOCUS: MODERATE DEMENTIA
# ============================================================

print("\n" + "=" * 75)
print("MODERATE DEMENTIA - DETAILED CHECK")
print("=" * 75)


moderate_subjects = subject_image_counts[
    "Moderate Dementia"
]


if len(moderate_subjects) == 0:

    print(
        "No Moderate Dementia subjects found."
    )

else:

    print(
        "Number of subjects:",
        len(moderate_subjects)
    )


    for subject_id, count in sorted(
        moderate_subjects.items()
    ):

        print(
            f"{subject_id:15s} : "
            f"{count:4d} images"
        )


# ============================================================
# FINAL TOTAL
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)


print(
    "Total images scanned:",
    total_images
)


print(
    "Total unique subjects:"
)


all_subjects = set()


for class_name in class_names:

    all_subjects.update(

        subject_image_counts[
            class_name
        ].keys()

    )


print(
    len(all_subjects)
)


print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)