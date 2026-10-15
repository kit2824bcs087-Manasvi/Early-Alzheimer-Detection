import os
import csv
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image


# ============================================================
# FINAL ALZHEIMER MRI MODEL
# ============================================================

CSV_FILE = "cv_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8

# Conservative final training
EPOCHS = 3

LEARNING_RATE = 0.00002
NUM_WORKERS = 0

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = 4

MODEL_DIR = os.path.join(
    "models",
    "final"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("       ALZHEIMER MRI FINAL MODEL")
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
# TRAIN TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
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
# LOAD ALL DATA
# ============================================================

def load_all_records():

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

            records.append(row)

    return records


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def calculate_class_weights(
    records
):

    counter = Counter(
        row["class"]
        for row in records
    )

    print(
        "\nDataset class distribution:"
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:25s}: "
            f"{counter[class_name]}"
        )

    # --------------------------------------------------------
    # Square-root inverse frequency
    # --------------------------------------------------------

    weights = []

    for class_name in CLASS_NAMES:

        count = counter[class_name]

        weight = 1.0 / (
            count ** 0.5
        )

        weights.append(
            weight
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    weights = (
        weights
        /
        weights.mean()
    )

    print(
        "\nClass weights:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:25s}: "
            f"{weights[index].item():.4f}"
        )

    return weights


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        inputs,
        targets
    ):

        log_prob = nn.functional.log_softmax(
            inputs,
            dim=1
        )

        prob = torch.exp(
            log_prob
        )

        target_log_prob = log_prob.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        target_prob = prob.gather(
            1,
            targets.unsqueeze(1)
        ).squeeze(1)

        focal_factor = (
            1.0 - target_prob
        ) ** self.gamma

        loss = (
            -target_log_prob
            *
            focal_factor
        )

        if self.alpha is not None:

            loss = (
                loss
                *
                self.alpha[targets]
            )

        return loss.mean()


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print(
        "\nLoading pretrained MobileNetV2..."
    )

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
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

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

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

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("LOADING COMPLETE DATASET")
    print("=" * 70)

    records = load_all_records()

    print(
        "\nTotal images:",
        len(records)
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = AlzheimerDataset(

        records,

        train_transform

    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        records
    )

    class_weights = class_weights.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Weighted sampler
    # --------------------------------------------------------

    sample_weights = []

    for row in records:

        label = CLASS_NAMES.index(
            row["class"]
        )

        sample_weights.append(
            class_weights[label].item()
        )

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(
            sample_weights
        ),

        replacement=True

    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        sampler=sampler,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available()

    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = FocalLoss(

        alpha=class_weights,

        gamma=2.0

    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-4

    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_loss = float("inf")

    best_epoch = 0

    print("\n")
    print("=" * 70)
    print("STARTING FINAL MODEL TRAINING")
    print("=" * 70)

    print(
        "\nModel          : MobileNetV2"
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
        EPOCHS
    )

    print(
        "Learning rate  :",
        LEARNING_RATE
    )

    print(
        "Device         :",
        DEVICE
    )

    print("=" * 70)

    for epoch in range(
        EPOCHS
    ):

        print("\n")
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 70)

        train_loss, train_accuracy = (
            train_one_epoch(

                model,

                loader,

                criterion,

                optimizer

            )
        )

        print("\nResults:")

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy

        })

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(

            MODEL_DIR,

            f"final_epoch{epoch + 1}.pth"

        )

        torch.save({

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "class_names":
                CLASS_NAMES,

            "image_size":
                IMAGE_SIZE

        }, checkpoint_path)

        print(
            "\n✓ Checkpoint saved:"
        )

        print(
            checkpoint_path
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if train_loss < best_loss:

            best_loss = train_loss

            best_epoch = (
                epoch + 1
            )

            best_path = os.path.join(

                MODEL_DIR,

                "alzheimer_mobilenetv2_final.pth"

            )

            torch.save({

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "image_size":
                    IMAGE_SIZE

            }, best_path)

            print(
                "✓ FINAL MODEL UPDATED"
            )

            print(
                "Path:",
                best_path
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_path = os.path.join(

        MODEL_DIR,

        "final_training_history.csv"

    )

    with open(

        history_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[

                "epoch",

                "train_loss",

                "train_accuracy"

            ]

        )

        writer.writeheader()

        writer.writerows(
            history
        )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        "FINAL MODEL TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        "\nBest epoch:",
        best_epoch
    )

    print(
        "\nFinal model:"
    )

    print(
        os.path.join(
            MODEL_DIR,
            "alzheimer_mobilenetv2_final.pth"
        )
    )

    print(
        "\nTraining history:"
    )

    print(
        history_path
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()