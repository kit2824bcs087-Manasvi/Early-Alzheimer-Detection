import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

CSV_FILE = "dataset_split.csv"

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 10

LEARNING_RATE = 0.0001

# Windows-la multiprocessing issue avoid panna 0
NUM_WORKERS = 0

MODEL_DIR = "models"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv2_alzheimer_best.pth"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "training_history.csv"
)

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("       ALZHEIMER MRI CLASSIFICATION")
    print("       MobileNetV2 Training")
    print("=" * 70)


    # ========================================================
    # 4. DEVICE / GPU
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

        print("\nWARNING: CUDA not available.")
        print("Training will use CPU.")


    print(
        "\nDevice:",
        device
    )


    # ========================================================
    # 5. CREATE MODEL DIRECTORY
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )


    # ========================================================
    # 6. CHECK CSV FILE
    # ========================================================

    if not os.path.exists(CSV_FILE):

        print("\nERROR:")
        print(
            "dataset_split.csv was not found."
        )

        print(
            "Make sure dataset_split.csv "
            "is in the same folder as train.py."
        )

        return


    # ========================================================
    # 7. READ DATASET SPLIT
    # ========================================================

    train_records = []

    validation_records = []

    test_records = []


    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            record = {

                "image_path":
                    row["image_path"],

                "subject_id":
                    row["subject_id"],

                "class":
                    row["class"]

            }


            if row["split"] == "train":

                train_records.append(
                    record
                )

            elif row["split"] == "validation":

                validation_records.append(
                    record
                )

            elif row["split"] == "test":

                test_records.append(
                    record
                )


    # ========================================================
    # 8. DATASET INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        "Training images   :",
        len(train_records)
    )

    print(
        "Validation images :",
        len(validation_records)
    )

    print(
        "Test images       :",
        len(test_records)
    )


    # ========================================================
    # 9. IMAGE TRANSFORMS
    # ========================================================

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


    # ========================================================
    # 10. CREATE DATASETS
    # ========================================================

    train_dataset = AlzheimerDataset(
        train_records,
        train_transform
    )

    validation_dataset = AlzheimerDataset(
        validation_records,
        validation_transform
    )

    test_dataset = AlzheimerDataset(
        test_records,
        validation_transform
    )


    # ========================================================
    # 11. CREATE DATALOADERS
    # ========================================================

    # IMPORTANT:
    # num_workers = 0
    #
    # This avoids Windows multiprocessing
    # problems.

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # 12. LOAD MOBILENETV2
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING MOBILENETV2")
    print("=" * 70)


    try:

        weights = (
            models.MobileNet_V2_Weights.DEFAULT
        )

        model = models.mobilenet_v2(
            weights=weights
        )

        print(
            "Pretrained ImageNet weights loaded."
        )

    except Exception as error:

        print(
            "Could not load pretrained weights."
        )

        print(
            "Reason:",
            error
        )

        print(
            "Using MobileNetV2 without "
            "pretrained weights."
        )

        model = models.mobilenet_v2(
            weights=None
        )


    # ========================================================
    # 13. MODIFY FINAL CLASSIFIER
    # ========================================================

    input_features = (
        model.classifier[1].in_features
    )


    model.classifier[1] = nn.Linear(

        input_features,

        NUM_CLASSES
    )


    model = model.to(device)


    # ========================================================
    # 14. CLASS WEIGHTS
    # ========================================================

    # Calculated from training data
    #
    # Mild Dementia       : 3294
    # Moderate Dementia   : 244
    # Non Demented        : 47153
    # Very mild Dementia  : 9333

    class_weights = torch.tensor(

        [
            4.555555,
            61.500000,
            0.318241,
            1.607843
        ],

        dtype=torch.float32,

        device=device
    )


    print("\nClass weights:")

    for index in range(NUM_CLASSES):

        print(

            f"{CLASS_NAMES[index]:25s}: "
            f"{class_weights[index].item():.6f}"

        )


    # ========================================================
    # 15. LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss(

        weight=class_weights
    )


    # ========================================================
    # 16. OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # ========================================================
    # 17. TRAIN FUNCTION
    # ========================================================

    def train_one_epoch():

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0


        for batch_index, (
            images,
            labels
        ) in enumerate(train_loader):


            images = images.to(

                device,

                non_blocking=True
            )


            labels = labels.to(

                device,

                non_blocking=True
            )


            # Clear previous gradients

            optimizer.zero_grad()


            # Forward pass

            outputs = model(images)


            # Calculate loss

            loss = criterion(

                outputs,

                labels
            )


            # Backpropagation

            loss.backward()


            # Update weights

            optimizer.step()


            # Statistics

            total_loss += (

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


            # Progress

            if (
                batch_index + 1
            ) % 500 == 0:

                print(

                    f"    Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)}"

                )


        average_loss = (

            total_loss / total
        )


        accuracy = (

            correct / total
        ) * 100


        return (
            average_loss,
            accuracy
        )


    # ========================================================
    # 18. VALIDATION FUNCTION
    # ========================================================

    def validate():

        model.eval()

        total_loss = 0.0

        correct = 0

        total = 0


        with torch.no_grad():

            for images, labels in validation_loader:


                images = images.to(

                    device,

                    non_blocking=True
                )


                labels = labels.to(

                    device,

                    non_blocking=True
                )


                outputs = model(images)


                loss = criterion(

                    outputs,

                    labels
                )


                total_loss += (

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


        average_loss = (

            total_loss / total
        )


        accuracy = (

            correct / total
        ) * 100


        return (
            average_loss,
            accuracy
        )


    # ========================================================
    # 19. START TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        "Model          : MobileNetV2"
    )

    print(
        "Image size     :",
        IMAGE_SIZE
    )

    print(
        "Batch size     :",
        BATCH_SIZE
    )

    print(
        "Epochs         :",
        NUM_EPOCHS
    )

    print(
        "Learning rate  :",
        LEARNING_RATE
    )

    print(
        "DataLoader workers:",
        0
    )

    print(
        "Device         :",
        device
    )

    print("=" * 70)


    best_validation_loss = float(
        "inf"
    )


    history = []


    # ========================================================
    # 20. EPOCH LOOP
    # ========================================================

    for epoch in range(NUM_EPOCHS):


        print(

            f"\nEpoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"

        )

        print(
            "-" * 70
        )


        # Training

        train_loss, train_accuracy = (
            train_one_epoch()
        )


        # Validation

        validation_loss, validation_accuracy = (
            validate()
        )


        # Results

        print("\nResults:")

        print(

            f"Train Loss      : "
            f"{train_loss:.4f}"

        )

        print(

            f"Train Accuracy  : "
            f"{train_accuracy:.2f}%"

        )

        print(

            f"Validation Loss : "
            f"{validation_loss:.4f}"

        )

        print(

            f"Validation Acc  : "
            f"{validation_accuracy:.2f}%"

        )


        # Save history

        history.append([

            epoch + 1,

            train_loss,

            train_accuracy,

            validation_loss,

            validation_accuracy

        ])


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            validation_loss
            <
            best_validation_loss
        ):


            best_validation_loss = (
                validation_loss
            )


            torch.save(

                {

                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "validation_loss":
                        validation_loss,

                    "class_names":
                        CLASS_NAMES,

                    "image_size":
                        IMAGE_SIZE,

                    "class_weights":
                        class_weights.cpu()

                },

                BEST_MODEL_PATH
            )


            print(
                "\n✓ BEST MODEL SAVED"
            )

            print(
                "Path:",
                BEST_MODEL_PATH
            )


    # ========================================================
    # 21. SAVE TRAINING HISTORY
    # ========================================================

    with open(

        HISTORY_FILE,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:


        writer = csv.writer(file)


        writer.writerow([

            "epoch",

            "train_loss",

            "train_accuracy",

            "validation_loss",

            "validation_accuracy"

        ])


        writer.writerows(history)


    # ========================================================
    # 22. TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nTraining history:"
    )

    print(
        HISTORY_FILE
    )

    print("=" * 70)


# ============================================================
# 23. DATASET CLASS
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


    def __getitem__(
        self,
        index
    ):

        record = self.records[index]


        image_path = (
            record["image_path"]
        )


        class_name = (
            record["class"]
        )


        label = CLASS_NAMES.index(
            class_name
        )


        try:

            image = Image.open(

                image_path

            ).convert("RGB")


        except Exception as error:

            print(
                "\nERROR loading image:"
            )

            print(
                image_path
            )

            print(
                error
            )

            raise


        if self.transform:

            image = self.transform(
                image
            )


        return (
            image,
            label
        )


# ============================================================
# 24. RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main() 

    