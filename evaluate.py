import csv
import os

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

CSV_FILE = "dataset_split.csv"

MODEL_PATH = "models/mobilenetv2_alzheimer_best.pth"

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
# DATASET
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(self, records, transform=None):

        self.records = records
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = record["image_path"]

        label = CLASS_NAMES.index(
            record["class"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("       ALZHEIMER MRI MODEL EVALUATION")
    print("=" * 70)


    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():

        device = torch.device("cuda")

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

        device = torch.device("cpu")

        print("\nUsing CPU")


    print(
        "\nDevice:",
        device
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(CSV_FILE):

        print(
            "\nERROR: dataset_split.csv not found."
        )

        return


    if not os.path.exists(MODEL_PATH):

        print(
            "\nERROR: Best model not found:"
        )

        print(
            MODEL_PATH
        )

        return


    # ========================================================
    # READ TEST DATA
    # ========================================================

    test_records = []


    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["split"] == "test":

                test_records.append({

                    "image_path":
                        row["image_path"],

                    "subject_id":
                        row["subject_id"],

                    "class":
                        row["class"]

                })


    print("\n" + "=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print(
        "Test images:",
        len(test_records)
    )


    # ========================================================
    # TRANSFORM
    # ========================================================

    test_transform = transforms.Compose([

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


    # ========================================================
    # DATASET + DATALOADER
    # ========================================================

    test_dataset = AlzheimerDataset(

        test_records,

        test_transform
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # LOAD MOBILENETV2
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)


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


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(device)

    model.eval()


    print(
        "Best checkpoint loaded successfully."
    )

    print(
        "Checkpoint epoch:",
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


    # ========================================================
    # PREDICTION
    # ========================================================

    all_labels = []

    all_predictions = []


    print("\n" + "=" * 70)
    print("RUNNING TEST EVALUATION")
    print("=" * 70)


    with torch.no_grad():

        for batch_index, (
            images,
            labels
        ) in enumerate(test_loader):


            images = images.to(

                device,

                non_blocking=True
            )


            outputs = model(images)


            predictions = outputs.argmax(

                dim=1
            )


            all_labels.extend(

                labels.numpy().tolist()
            )


            all_predictions.extend(

                predictions.cpu().numpy().tolist()
            )


            if (
                batch_index + 1
            ) % 500 == 0:

                print(

                    f"    Batch "
                    f"{batch_index + 1}/"
                    f"{len(test_loader)}"

                )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )


    # ========================================================
    # PRECISION
    # ========================================================

    precision = precision_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0
    )


    # ========================================================
    # RECALL
    # ========================================================

    recall = recall_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0
    )


    # ========================================================
    # F1 SCORE
    # ========================================================

    f1 = f1_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        all_labels,

        all_predictions,

        labels=list(
            range(NUM_CLASSES)
        )
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)


    print(
        f"\nAccuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1-score  : {f1 * 100:.2f}%"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)


    report = classification_report(

        all_labels,

        all_predictions,

        labels=list(
            range(NUM_CLASSES)
        ),

        target_names=CLASS_NAMES,

        zero_division=0
    )


    print(report)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)


    print(
        "\nRows    = Actual"
    )

    print(
        "Columns = Predicted\n"
    )


    print(
        " " * 22,
        CLASS_NAMES
    )


    for i in range(NUM_CLASSES):

        print(

            f"{CLASS_NAMES[i]:22s}",

            cm[i].tolist()

        )


    # ========================================================
    # SENSITIVITY + SPECIFICITY
    # ========================================================

    print("\n" + "=" * 70)
    print("SENSITIVITY AND SPECIFICITY")
    print("=" * 70)


    for i in range(NUM_CLASSES):

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
            cm.sum()
            - true_positive
            - false_negative
            - false_positive
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

            f"\n{CLASS_NAMES[i]}"

        )

        print(

            f"  Sensitivity : "
            f"{sensitivity * 100:.2f}%"

        )

        print(

            f"  Specificity : "
            f"{specificity * 100:.2f}%"

        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_file = (
        "models/test_results.txt"
    )


    with open(

        results_file,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(
            "ALZHEIMER MRI CLASSIFICATION\n"
        )

        file.write(
            "TEST SET EVALUATION\n\n"
        )


        file.write(

            f"Test images: "
            f"{len(test_records)}\n\n"

        )


        file.write(

            f"Accuracy: "
            f"{accuracy * 100:.2f}%\n"

        )

        file.write(

            f"Precision: "
            f"{precision * 100:.2f}%\n"

        )

        file.write(

            f"Recall: "
            f"{recall * 100:.2f}%\n"

        )

        file.write(

            f"F1-score: "
            f"{f1 * 100:.2f}%\n\n"

        )


        file.write(
            "Classification Report:\n"
        )

        file.write(
            report
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )


    print("\n" + "=" * 70)

    print(
        "EVALUATION COMPLETED"
    )

    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(
        results_file
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()