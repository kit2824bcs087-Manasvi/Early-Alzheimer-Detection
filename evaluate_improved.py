import os
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "cv_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

MODEL_DIR = os.path.join(
    "models",
    "improved"
)

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

print("=" * 70)
print("       ALZHEIMER MRI IMPROVED MODEL EVALUATION")
print("=" * 70)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("\nGPU DETECTED")
    print("-" * 70)

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

    print("\nCPU MODE")

print(
    "\nDevice:",
    DEVICE
)


# ============================================================
# DATASET
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(
        self,
        records,
        transform=None
    ):

        self.records = records
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        row = self.records[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        label = CLASS_NAMES.index(
            row["class"]
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

validation_transform = transforms.Compose([

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
# LOAD VALIDATION RECORDS
# ============================================================

def load_validation_records(
    fold
):

    records = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            if int(row["fold"]) != fold:
                continue

            if row["split"] == "validation":

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

    model.classifier[1] = nn.Sequential(

        nn.Dropout(
            p=0.4
        ),

        nn.Linear(
            input_features,
            NUM_CLASSES
        )

    )

    return model


# ============================================================
# LOAD BEST MODEL
# ============================================================

def load_model(
    fold
):

    model_path = os.path.join(

        MODEL_DIR,

        f"mobilenetv2_improved_fold{fold}_best.pth"

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

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{model_path}"
        )

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
            "Unknown"
        )
    )

    print(
        "Validation loss:",
        checkpoint.get(
            "validation_loss",
            "Unknown"
        )
    )

    print(
        "Validation accuracy:",
        checkpoint.get(
            "validation_accuracy",
            "Unknown"
        )
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_fold(
    fold
):

    print("\n")
    print("=" * 70)

    print(
        f"IMPROVED MODEL - FOLD {fold} EVALUATION"
    )

    print("=" * 70)

    records = load_validation_records(
        fold
    )

    print(
        "\nValidation images:",
        len(records)
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in CLASS_NAMES
    }

    for row in records:

        class_counts[
            row["class"]
        ] += 1

    print(
        "\nValidation class distribution:"
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:25s}: "
            f"{class_counts[class_name]}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = AlzheimerDataset(

        records,

        validation_transform

    )

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        fold
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_predictions = []

    all_labels = []

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

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
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
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            average="weighted",

            zero_division=0

        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            average="macro",

            zero_division=0

        )
    )

    print("\n")
    print("=" * 70)
    print(
        f"FOLD {fold} IMPROVED RESULTS"
    )
    print("=" * 70)

    print(
        f"\nAccuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision: "
        f"{weighted_precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall   : "
        f"{weighted_recall * 100:.2f}%"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1 * 100:.2f}%"
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
    # Classification report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)

    report = classification_report(

        all_labels,

        all_predictions,

        target_names=CLASS_NAMES,

        zero_division=0

    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(

        all_labels,

        all_predictions

    )

    print("\n")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nRows    = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print(
        "\nClasses:"
    )

    print(
        CLASS_NAMES
    )

    for index, row in enumerate(cm):

        print(
            f"{CLASS_NAMES[index]:25s}",
            list(row)
        )

    # --------------------------------------------------------
    # Sensitivity and specificity
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "SENSITIVITY AND SPECIFICITY"
    )
    print("=" * 70)

    total_samples = cm.sum()

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        true_positive = cm[i, i]

        false_negative = (
            cm[i, :].sum()
            - true_positive
        )

        false_positive = (
            cm[:, i].sum()
            - true_positive
        )

        true_negative = (
            total_samples
            - true_positive
            - false_positive
            - false_negative
        )

        sensitivity = (

            true_positive
            /
            (
                true_positive
                +
                false_negative
            )

            if (
                true_positive
                +
                false_negative
            ) > 0

            else 0

        )

        specificity = (

            true_negative
            /
            (
                true_negative
                +
                false_positive
            )

            if (
                true_negative
                +
                false_positive
            ) > 0

            else 0

        )

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
    # Save results
    # --------------------------------------------------------

    result_path = os.path.join(

        MODEL_DIR,

        f"improved_fold{fold}_evaluation.txt"

    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ALZHEIMER MRI "
            "IMPROVED MODEL EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Fold: {fold}\n"
        )

        file.write(
            f"Validation images: "
            f"{len(records)}\n\n"
        )

        file.write(
            f"Accuracy: "
            f"{accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Weighted Precision: "
            f"{weighted_precision * 100:.2f}%\n"
        )

        file.write(
            f"Weighted Recall: "
            f"{weighted_recall * 100:.2f}%\n"
        )

        file.write(
            f"Weighted F1: "
            f"{weighted_f1 * 100:.2f}%\n\n"
        )

        file.write(
            f"Macro Precision: "
            f"{macro_precision * 100:.2f}%\n"
        )

        file.write(
            f"Macro Recall: "
            f"{macro_recall * 100:.2f}%\n"
        )

        file.write(
            f"Macro F1: "
            f"{macro_f1 * 100:.2f}%\n\n"
        )

        file.write(
            "PER-CLASS PERFORMANCE\n"
        )

        file.write(
            "-" * 70 + "\n"
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
        result_path
    )

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    fold_results = []

    # --------------------------------------------------------
    # Fold 1
    # --------------------------------------------------------

    result1 = evaluate_fold(
        1
    )

    fold_results.append(
        result1
    )

    # --------------------------------------------------------
    # Fold 2
    # --------------------------------------------------------

    result2 = evaluate_fold(
        2
    )

    fold_results.append(
        result2
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    mean_accuracy = sum(
        result["accuracy"]
        for result in fold_results
    ) / len(fold_results)

    mean_weighted_f1 = sum(
        result["weighted_f1"]
        for result in fold_results
    ) / len(fold_results)

    mean_macro_f1 = sum(
        result["macro_f1"]
        for result in fold_results
    ) / len(fold_results)

    print("\n")
    print("=" * 70)
    print(
        "IMPROVED MODEL 2-FOLD CV SUMMARY"
    )
    print("=" * 70)

    print(
        "\nFold 1"
    )

    print(
        f"  Accuracy    : "
        f"{result1['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Weighted F1 : "
        f"{result1['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"  Macro F1    : "
        f"{result1['macro_f1'] * 100:.2f}%"
    )

    print(
        "\nFold 2"
    )

    print(
        f"  Accuracy    : "
        f"{result2['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Weighted F1 : "
        f"{result2['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"  Macro F1    : "
        f"{result2['macro_f1'] * 100:.2f}%"
    )

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

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = os.path.join(

        MODEL_DIR,

        "improved_cv_summary.txt"

    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "IMPROVED MODEL 2-FOLD CV SUMMARY\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Fold 1 Accuracy: "
            f"{result1['accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Fold 1 Weighted F1: "
            f"{result1['weighted_f1'] * 100:.2f}%\n"
        )

        file.write(
            f"Fold 1 Macro F1: "
            f"{result1['macro_f1'] * 100:.2f}%\n\n"
        )

        file.write(
            f"Fold 2 Accuracy: "
            f"{result2['accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Fold 2 Weighted F1: "
            f"{result2['weighted_f1'] * 100:.2f}%\n"
        )

        file.write(
            f"Fold 2 Macro F1: "
            f"{result2['macro_f1'] * 100:.2f}%\n\n"
        )

        file.write(
            f"Mean Accuracy: "
            f"{mean_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Mean Weighted F1: "
            f"{mean_weighted_f1 * 100:.2f}%\n"
        )

        file.write(
            f"Mean Macro F1: "
            f"{mean_macro_f1 * 100:.2f}%\n"
        )

    print(
        "\nSummary saved:"
    )

    print(
        summary_path
    )

    print("\n")
    print("=" * 70)
    print(
        "IMPROVED MODEL EVALUATION COMPLETED"
    )
    print("=" * 70)


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()