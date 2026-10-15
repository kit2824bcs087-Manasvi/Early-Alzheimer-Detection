import os
import csv
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image


# ============================================================
# ALZHEIMER MRI - IMPROVED MODEL TRAINING
# ============================================================

CSV_FILE = "cv_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.00005
NUM_WORKERS = 0

PATIENCE = 3

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = 4

MODEL_DIR = os.path.join(
    "models",
    "improved"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("       ALZHEIMER MRI CLASSIFICATION")
print("       IMPROVED MODEL EXPERIMENT")
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
# RANDOM SEED
# ============================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# DATASET CLASS
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

        image_path = row["image_path"]

        image = Image.open(
            image_path
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
# TRAINING AUGMENTATION
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

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
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

            alpha_factor = self.alpha[
                targets
            ]

            loss = (
                loss
                *
                alpha_factor
            )

        return loss.mean()


# ============================================================
# LOAD CV DATA
# ============================================================

def load_fold_records(fold):

    train_records = []
    validation_records = []

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

            if row["split"] == "train":

                train_records.append(
                    row
                )

            elif row["split"] == "validation":

                validation_records.append(
                    row
                )

    return (
        train_records,
        validation_records
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def get_class_counts(
    records
):

    counter = Counter(
        row["class"]
        for row in records
    )

    return [
        counter[class_name]
        for class_name in CLASS_NAMES
    ]


# ============================================================
# CREATE BALANCED SAMPLER
# ============================================================

def create_sampler(
    records
):

    class_counts = get_class_counts(
        records
    )

    print(
        "\nTraining class distribution:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:25s}: "
            f"{class_counts[index]}"
        )

    # --------------------------------------------------------
    # Inverse square-root frequency
    #
    # Less aggressive than direct inverse frequency.
    # Helps minority classes without excessively duplicating
    # the very small Moderate Dementia class.
    # --------------------------------------------------------

    class_weights = []

    for count in class_counts:

        weight = 1.0 / (
            count ** 0.5
        )

        class_weights.append(
            weight
        )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    # Normalize average weight to 1

    class_weights = (
        class_weights
        /
        class_weights.mean()
    )

    print(
        "\nSampler class weights:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:25s}: "
            f"{class_weights[index].item():.4f}"
        )

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

    return (
        sampler,
        class_weights
    )


# ============================================================
# CREATE MOBILENETV2
# ============================================================

def create_model():

    print(
        "\nLoading MobileNetV2..."
    )

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    input_features = (
        model.classifier[1].in_features
    )

    # --------------------------------------------------------
    # Dropout + classifier
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


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

            outputs = model(
                images
            )

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
    fold
):

    print("\n")
    print("=" * 70)
    print(
        f"IMPROVED MODEL - FOLD {fold}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    (
        train_records,
        validation_records
    ) = load_fold_records(
        fold
    )

    print(
        "\nTraining images   :",
        len(train_records)
    )

    print(
        "Validation images :",
        len(validation_records)
    )

    # --------------------------------------------------------
    # Dataset
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
    # Balanced sampler
    # --------------------------------------------------------

    (
        sampler,
        sampler_weights
    ) = create_sampler(
        train_records
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        sampler=sampler,

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
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Focal Loss alpha
    #
    # Use a mild version of class weighting.
    # This avoids the extreme 61.5 weight from the original
    # experiment.
    # --------------------------------------------------------

    alpha = sampler_weights.clone()

    # Normalize alpha around 1

    alpha = (
        alpha
        /
        alpha.mean()
    )

    alpha = alpha.to(
        DEVICE
    )

    print(
        "\nFocal Loss alpha:"
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name:25s}: "
            f"{alpha[index].item():.4f}"
        )

    criterion = FocalLoss(

        alpha=alpha,

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
    # Scheduler
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=1,

            min_lr=1e-7

        )
    )

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    patience_counter = 0

    history = []

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        print("\n")
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy
        ) = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer

        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy
        ) = validate(

            model,

            validation_loader,

            criterion

        )

        current_lr = (
            optimizer
            .param_groups[0]
            ["lr"]
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

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
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Acc  : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate   : "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "validation_loss":
                val_loss,

            "validation_accuracy":
                val_accuracy,

            "learning_rate":
                current_lr

        })

        # ----------------------------------------------------
        # Save EVERY epoch
        # ----------------------------------------------------

        epoch_path = os.path.join(

            MODEL_DIR,

            f"improved_fold{fold}_epoch{epoch + 1}.pth"

        )

        torch.save({

            "fold":
                fold,

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "validation_loss":
                val_loss,

            "validation_accuracy":
                val_accuracy,

            "class_names":
                CLASS_NAMES

        }, epoch_path)

        print(
            "\n✓ Checkpoint saved:"
        )

        print(
            epoch_path
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            best_path = os.path.join(

                MODEL_DIR,

                f"mobilenetv2_improved_fold{fold}_best.pth"

            )

            torch.save({

                "fold":
                    fold,

                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "validation_loss":
                    val_loss,

                "validation_accuracy":
                    val_accuracy,

                "class_names":
                    CLASS_NAMES

            }, best_path)

            print(
                "✓ BEST MODEL UPDATED"
            )

            print(
                "Path:",
                best_path
            )

        else:

            patience_counter += 1

            print(
                f"No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_path = os.path.join(

            MODEL_DIR,

            f"improved_fold{fold}_history.csv"

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

                    "train_accuracy",

                    "validation_loss",

                    "validation_accuracy",

                    "learning_rate"

                ]

            )

            writer.writeheader()

            writer.writerows(
                history
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= PATIENCE
        ):

            print("\n")
            print(
                "EARLY STOPPING"
            )

            print(
                "Validation loss did not "
                "improve for "
                f"{PATIENCE} epochs."
            )

            break

    # ========================================================
    # FOLD COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        f"IMPROVED FOLD {fold} COMPLETED"
    )

    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Best model:"
    )

    print(
        os.path.join(
            MODEL_DIR,
            f"mobilenetv2_improved_fold{fold}_best.pth"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print(
        "Method:"
    )

    print(
        "MobileNetV2"
    )

    print(
        "+ Data Augmentation"
    )

    print(
        "+ Dropout"
    )

    print(
        "+ Weighted Sampling"
    )

    print(
        "+ Focal Loss"
    )

    print(
        "+ AdamW"
    )

    print(
        "+ Learning Rate Scheduler"
    )

    print(
        "+ Early Stopping"
    )

    print(
        "\nOutput directory:"
    )

    print(
        MODEL_DIR
    )

    # ========================================================
    # FOLD 1
    # ========================================================

    train_fold(1)

    # ========================================================
    # FOLD 2
    # ========================================================

    train_fold(2)

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "IMPROVED MODEL TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        "\nModels saved inside:"
    )

    print(
        MODEL_DIR
    )


# ============================================================
# WINDOWS SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()