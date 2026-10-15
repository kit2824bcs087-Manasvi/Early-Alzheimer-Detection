import os
import csv
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "cv_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

NUM_CLASSES = 4

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

MODEL_DIR = "models/cv"


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
# DATASET CLASS
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(self, records, transform=None):

        self.records = records
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        label = CLASS_NAMES.index(
            record["class"]
        )

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# LOAD CSV
# ============================================================

def load_records():

    fold_records = {
        1: {
            "train": [],
            "validation": []
        },

        2: {
            "train": [],
            "validation": []
        }
    }


    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            fold = int(row["fold"])

            split = row["split"]

            fold_records[fold][split].append(row)


    return fold_records


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
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
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(records):

    counts = Counter()

    for record in records:

        label = CLASS_NAMES.index(
            record["class"]
        )

        counts[label] += 1


    total = sum(
        counts.values()
    )


    weights = []


    for i in range(NUM_CLASSES):

        count = counts[i]

        if count == 0:

            weight = 0.0

        else:

            weight = total / (
                NUM_CLASSES * count
            )

        weights.append(weight)


    return weights, counts


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(loader):


        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        if (
            batch_index + 1
        ) % 500 == 0:

            print(
                f"    Batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    validation_loss = (
        running_loss / total
    )

    validation_accuracy = (
        correct / total
    )


    return (
        validation_loss,
        validation_accuracy
    )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold,
    train_records,
    validation_records
):

    print("\n")
    print("=" * 70)
    print(
        f"STARTING FOLD {fold}"
    )
    print("=" * 70)


    print(
        "\nTraining images:",
        len(train_records)
    )

    print(
        "Validation images:",
        len(validation_records)
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nTraining distribution:")

    train_counts = Counter(
        r["class"]
        for r in train_records
    )


    for class_name in CLASS_NAMES:

        print(
            f"{class_name:25s}: "
            f"{train_counts[class_name]}"
        )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = AlzheimerDataset(
        train_records,
        train_transform
    )


    validation_dataset = AlzheimerDataset(
        validation_records,
        validation_transform
    )


    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading MobileNetV2...")

    model = create_model()

    model = model.to(DEVICE)


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    weights, counts = calculate_class_weights(
        train_records
    )


    print("\nClass weights:")

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:25s}: "
            f"{weights[i]:.6f}"
        )


    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    best_validation_loss = float(
        "inf"
    )


    history = []


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(EPOCHS):


        print("\n")
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 70)


        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )


        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion
            )
        )


        print("\nResults:")

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss : "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Acc  : "
            f"{validation_accuracy * 100:.2f}%"
        )


        history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "validation_loss":
                validation_loss,

            "validation_accuracy":
                validation_accuracy

        })


        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )


            model_path = os.path.join(

                MODEL_DIR,

                f"mobilenetv2_fold{fold}_best.pth"

            )


            torch.save({

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "validation_loss":
                    validation_loss,

                "validation_accuracy":
                    validation_accuracy,

                "class_names":
                    CLASS_NAMES

            }, model_path)


            print(
                "\n✓ BEST MODEL SAVED"
            )

            print(
                "Path:",
                model_path
            )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_file = os.path.join(

        MODEL_DIR,

        f"fold{fold}_history.csv"

    )


    with open(

        history_file,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy"
            ]

        )


        writer.writeheader()

        writer.writerows(history)


    print("\n")
    print(
        f"FOLD {fold} COMPLETED"
    )

    print(
        "Best validation loss:",
        best_validation_loss
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("       ALZHEIMER MRI 2-FOLD CROSS-VALIDATION")
    print("       MobileNetV2")
    print("=" * 70)


    # --------------------------------------------------------
    # CHECK CSV
    # --------------------------------------------------------

    if not os.path.exists(
        CSV_FILE
    ):

        print(
            "\nERROR:"
        )

        print(
            f"{CSV_FILE} not found."
        )

        return


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    fold_records = load_records()


    print("\nDataset loaded successfully.")


    # --------------------------------------------------------
    # FOLD 1
    # --------------------------------------------------------

    train_fold(

        fold=1,

        train_records=
            fold_records[1]["train"],

        validation_records=
            fold_records[1]["validation"]

    )


    # --------------------------------------------------------
    # FOLD 2
    # --------------------------------------------------------

    train_fold(

        fold=2,

        train_records=
            fold_records[2]["train"],

        validation_records=
            fold_records[2]["validation"]

    )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("       2-FOLD CROSS-VALIDATION COMPLETED")
    print("=" * 70)


    print("\nModels saved in:")

    print(
        "models/cv/"
    )


    print("\nExpected files:")

    print(
        "models/cv/mobilenetv2_fold1_best.pth"
    )

    print(
        "models/cv/mobilenetv2_fold2_best.pth"
    )

    print(
        "models/cv/fold1_history.csv"
    )

    print(
        "models/cv/fold2_history.csv"
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()