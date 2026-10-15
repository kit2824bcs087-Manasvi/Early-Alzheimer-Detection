import os
import csv
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "cv_split.csv"

MODEL_DIR = "models/cv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = 4


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("=" * 70)
    print("GPU DETECTED")
    print("=" * 70)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

else:

    DEVICE = torch.device("cpu")

    print("=" * 70)
    print("USING CPU")
    print("=" * 70)


print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(
        self,
        records,
        transform
    ):

        self.records = records
        self.transform = transform


    def __len__(self):

        return len(self.records)


    def __getitem__(self, index):

        row = self.records[index]

        image_path = row[
            "image_path"
        ]

        image = Image.open(
            image_path
        ).convert("RGB")


        label = CLASS_NAMES.index(
            row["class"]
        )


        image = self.transform(
            image
        )


        return image, label


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_records(fold):

    records = []


    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)


        for row in reader:

            row_fold = int(
                row["fold"]
            )


            if row_fold != fold:

                continue


            if row["split"] != "validation":

                continue


            records.append(row)


    return records


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.mobilenet_v2(
        weights=None
    )


    input_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )


    return model


# ============================================================
# EVALUATE ONE FOLD
# ============================================================

def evaluate_fold(fold):

    print("\n")
    print("=" * 70)

    print(
        f"FOLD {fold} EVALUATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    records = load_validation_records(
        fold
    )


    print(
        "\nValidation images:",
        len(records)
    )


    # --------------------------------------------------------
    # SUBJECT COUNT
    # --------------------------------------------------------

    subjects = set(
        row["subject_id"]
        for row in records
    )


    print(
        "Validation subjects:",
        len(subjects)
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nValidation class distribution:"
    )


    class_counts = Counter(
        row["class"]
        for row in records
    )


    for class_name in CLASS_NAMES:

        print(

            f"{class_name:25s}: "
            f"{class_counts[class_name]}"

        )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = AlzheimerDataset(
        records,
        transform
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )


    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_path = os.path.join(

        MODEL_DIR,

        f"mobilenetv2_fold{fold}_best.pth"

    )


    print(
        "\nLoading model:"
    )

    print(
        model_path
    )


    if not os.path.exists(
        model_path
    ):

        print(
            "\nERROR: Model not found!"
        )

        return None


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    checkpoint = torch.load(

        model_path,

        map_location=DEVICE

    )


    model = create_model()


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model = model.to(
        DEVICE
    )


    model.eval()


    print(
        "\nCheckpoint epoch:",
        checkpoint.get(
            "epoch",
            "N/A"
        )
    )


    print(
        "Validation loss:",
        checkpoint.get(
            "validation_loss",
            "N/A"
        )
    )


    print(
        "Validation accuracy:",
        checkpoint.get(
            "validation_accuracy",
            "N/A"
        )
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    all_labels = []

    all_predictions = []


    print(
        "\nRunning evaluation..."
    )


    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(loader):


            images = images.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            predictions = outputs.argmax(
                dim=1
            )


            all_labels.extend(
                labels.numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            if (
                batch_index + 1
            ) % 500 == 0:

                print(

                    f"    Batch "
                    f"{batch_index + 1}/"
                    f"{len(loader)}"

                )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(

        all_labels,

        all_predictions

    )


    precision = precision_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )


    macro_precision = precision_score(

        all_labels,

        all_predictions,

        average="macro",

        zero_division=0

    )


    macro_recall = recall_score(

        all_labels,

        all_predictions,

        average="macro",

        zero_division=0

    )


    macro_f1 = f1_score(

        all_labels,

        all_predictions,

        average="macro",

        zero_division=0

    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        f"FOLD {fold} RESULTS"
    )

    print("=" * 70)


    print(
        f"\nAccuracy          : "
        f"{accuracy * 100:.2f}%"
    )


    print(
        f"Weighted Precision: "
        f"{precision * 100:.2f}%"
    )


    print(
        f"Weighted Recall   : "
        f"{recall * 100:.2f}%"
    )


    print(
        f"Weighted F1       : "
        f"{f1 * 100:.2f}%"
    )


    print(
        f"\nMacro Precision   : "
        f"{macro_precision * 100:.2f}%"
    )


    print(
        f"Macro Recall      : "
        f"{macro_recall * 100:.2f}%"
    )


    print(
        f"Macro F1          : "
        f"{macro_f1 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "PER-CLASS PERFORMANCE"
    )

    print("=" * 70)


    report = classification_report(

        all_labels,

        all_predictions,

        target_names=CLASS_NAMES,

        zero_division=0

    )


    print(
        report
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(

        all_labels,

        all_predictions,

        labels=[
            0,
            1,
            2,
            3
        ]

    )


    print("\n")
    print("=" * 70)

    print(
        "CONFUSION MATRIX"
    )

    print("=" * 70)


    print(
        "\nRows    = Actual"
    )

    print(
        "Columns = Predicted\n"
    )


    print(
        "Classes:"
    )

    print(
        CLASS_NAMES
    )


    print()


    for i, row in enumerate(cm):

        print(

            f"{CLASS_NAMES[i]:25s} "
            f"{row.tolist()}"

        )


    # --------------------------------------------------------
    # SENSITIVITY / SPECIFICITY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "SENSITIVITY AND SPECIFICITY"
    )

    print("=" * 70)


    for i, class_name in enumerate(
        CLASS_NAMES
    ):


        true_positive = cm[i, i]


        false_negative = (
            cm[i, :].sum()
            -
            true_positive
        )


        false_positive = (
            cm[:, i].sum()
            -
            true_positive
        )


        true_negative = (
            cm.sum()
            -
            true_positive
            -
            false_negative
            -
            false_positive
        )


        if (
            true_positive
            +
            false_negative
        ) > 0:

            sensitivity = (

                true_positive
                /
                (
                    true_positive
                    +
                    false_negative
                )

            )

        else:

            sensitivity = 0.0


        if (
            true_negative
            +
            false_positive
        ) > 0:

            specificity = (

                true_negative
                /
                (
                    true_negative
                    +
                    false_positive
                )

            )

        else:

            specificity = 0.0


        print(
            f"\n{class_name}"
        )


        print(

            f"  Sensitivity : "
            f"{sensitivity * 100:.2f}%"

        )


        print(

            f"  Specificity : "
            f"{specificity * 100:.2f}%"

        )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result_file = os.path.join(

        MODEL_DIR,

        f"fold{fold}_evaluation.txt"

    )


    with open(

        result_file,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(
            f"FOLD {fold} EVALUATION\n"
        )

        file.write(
            "=" * 70
            +
            "\n"
        )


        file.write(
            f"Accuracy: "
            f"{accuracy * 100:.4f}%\n"
        )


        file.write(
            f"Weighted Precision: "
            f"{precision * 100:.4f}%\n"
        )


        file.write(
            f"Weighted Recall: "
            f"{recall * 100:.4f}%\n"
        )


        file.write(
            f"Weighted F1: "
            f"{f1 * 100:.4f}%\n"
        )


        file.write(
            f"Macro Precision: "
            f"{macro_precision * 100:.4f}%\n"
        )


        file.write(
            f"Macro Recall: "
            f"{macro_recall * 100:.4f}%\n"
        )


        file.write(
            f"Macro F1: "
            f"{macro_f1 * 100:.4f}%\n"
        )


        file.write(
            "\nCLASSIFICATION REPORT\n"
        )

        file.write(
            report
        )


        file.write(
            "\nCONFUSION MATRIX\n"
        )


        file.write(
            str(cm)
        )


    print(
        "\nResults saved:"
    )

    print(
        result_file
    )


    return {

        "fold": fold,

        "accuracy": accuracy,

        "weighted_precision": precision,

        "weighted_recall": recall,

        "weighted_f1": f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "       ALZHEIMER MRI 2-FOLD CV EVALUATION"
    )

    print("=" * 70)


    results = []


    # --------------------------------------------------------
    # FOLD 1
    # --------------------------------------------------------

    result1 = evaluate_fold(1)


    if result1:

        results.append(
            result1
        )


    # --------------------------------------------------------
    # FOLD 2
    # --------------------------------------------------------

    result2 = evaluate_fold(2)


    if result2:

        results.append(
            result2
        )


    # --------------------------------------------------------
    # CV SUMMARY
    # --------------------------------------------------------

    if len(results) == 2:

        print("\n")
        print("=" * 70)

        print(
            "2-FOLD CROSS-VALIDATION SUMMARY"
        )

        print("=" * 70)


        for result in results:

            print(

                f"\nFold {result['fold']}"

            )


            print(

                f"  Accuracy     : "
                f"{result['accuracy'] * 100:.2f}%"

            )


            print(

                f"  Weighted F1  : "
                f"{result['weighted_f1'] * 100:.2f}%"

            )


            print(

                f"  Macro F1     : "
                f"{result['macro_f1'] * 100:.2f}%"

            )


        mean_accuracy = sum(

            r["accuracy"]

            for r in results

        ) / len(results)


        mean_weighted_f1 = sum(

            r["weighted_f1"]

            for r in results

        ) / len(results)


        mean_macro_f1 = sum(

            r["macro_f1"]

            for r in results

        ) / len(results)


        print("\n")
        print("-" * 70)


        print(

            f"Mean Accuracy    : "
            f"{mean_accuracy * 100:.2f}%"

        )


        print(

            f"Mean Weighted F1 : "
            f"{mean_weighted_f1 * 100:.2f}%"

        )


        print(

            f"Mean Macro F1    : "
            f"{mean_macro_f1 * 100:.2f}%"

        )


        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        summary_file = os.path.join(

            MODEL_DIR,

            "cv_summary.txt"

        )


        with open(

            summary_file,

            "w",

            encoding="utf-8"

        ) as file:


            file.write(
                "2-FOLD CROSS-VALIDATION SUMMARY\n"
            )

            file.write(
                "=" * 70
                +
                "\n"
            )


            for result in results:

                file.write(

                    f"\nFold {result['fold']}\n"

                )

                file.write(

                    f"Accuracy: "
                    f"{result['accuracy'] * 100:.4f}%\n"

                )

                file.write(

                    f"Weighted F1: "
                    f"{result['weighted_f1'] * 100:.4f}%\n"

                )

                file.write(

                    f"Macro F1: "
                    f"{result['macro_f1'] * 100:.4f}%\n"

                )


            file.write(
                "\n"
            )


            file.write(

                f"Mean Accuracy: "
                f"{mean_accuracy * 100:.4f}%\n"

            )


            file.write(

                f"Mean Weighted F1: "
                f"{mean_weighted_f1 * 100:.4f}%\n"

            )


            file.write(

                f"Mean Macro F1: "
                f"{mean_macro_f1 * 100:.4f}%\n"

            )


        print(
            "\nSummary saved:"
        )

        print(
            summary_file
        )


    print("\n")
    print("=" * 70)

    print(
        "CV EVALUATION COMPLETED"
    )

    print("=" * 70)


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()