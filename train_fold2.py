import os
import csv
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "cv_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

CLASS_NAMES = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

NUM_CLASSES = 4

MODEL_DIR = "models/cv"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("=" * 70)
    print("GPU DETECTED")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

else:

    DEVICE = torch.device("cpu")

    print("=" * 70)
    print("CPU MODE")
    print("=" * 70)

print("\nDevice:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class AlzheimerDataset(Dataset):

    def __init__(self, records, transform):

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

        image = self.transform(image)

        return image, label


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
        10
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
# LOAD FOLD 2
# ============================================================

train_records = []
validation_records = []


with open(
    CSV_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if int(row["fold"]) != 2:
            continue

        if row["split"] == "train":

            train_records.append(row)

        elif row["split"] == "validation":

            validation_records.append(row)


print("\n" + "=" * 70)
print("FOLD 2 DATASET")
print("=" * 70)

print(
    "Training images:",
    len(train_records)
)

print(
    "Validation images:",
    len(validation_records)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining distribution:")

train_counts = Counter(
    row["class"]
    for row in train_records
)

for class_name in CLASS_NAMES:

    print(
        f"{class_name:25s}: "
        f"{train_counts[class_name]}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = AlzheimerDataset(
    train_records,
    train_transform
)

validation_dataset = AlzheimerDataset(
    validation_records,
    validation_transform
)


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


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILENETV2")
print("=" * 70)


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


model = model.to(DEVICE)


print(
    "Pretrained ImageNet weights loaded."
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

total = len(train_records)

class_weights = []


print("\nClass weights:")


for class_name in CLASS_NAMES:

    count = train_counts[class_name]

    weight = total / (
        NUM_CLASSES * count
    )

    class_weights.append(weight)

    print(
        f"{class_name:25s}: "
        f"{weight:.6f}"
    )


class_weights = torch.tensor(
    class_weights,
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


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total_images = 0


    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

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


        total_images += labels.size(0)


        if (
            batch_index + 1
        ) % 500 == 0:

            print(
                f"    Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )


    return (

        running_loss / total_images,

        correct / total_images

    )


def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total_images = 0


    with torch.no_grad():

        for images, labels in validation_loader:

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


            total_images += labels.size(0)


    return (

        running_loss / total_images,

        correct / total_images

    )


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING FOLD 2 TRAINING")
print("=" * 70)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)


best_val_loss = float("inf")

history = []


for epoch in range(EPOCHS):

    print("\n")
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("-" * 70)


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = validate()


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
            val_accuracy

    })


    # ========================================================
    # SAVE EVERY EPOCH
    # ========================================================

    checkpoint_path = os.path.join(

        MODEL_DIR,

        f"fold2_epoch{epoch + 1}.pth"

    )


    torch.save({

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

    }, checkpoint_path)


    print(
        "\n✓ Checkpoint saved:"
    )

    print(
        checkpoint_path
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        best_path = os.path.join(

            MODEL_DIR,

            "mobilenetv2_fold2_best.pth"

        )


        torch.save({

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


    # ========================================================
    # SAVE HISTORY AFTER EVERY EPOCH
    # ========================================================

    history_file = os.path.join(

        MODEL_DIR,

        "fold2_history.csv"

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


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FOLD 2 TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest validation loss:",
    best_val_loss
)

print(
    "\nBest model:"
)

print(
    "models/cv/mobilenetv2_fold2_best.pth"
)

print(
    "\nCheckpoints saved after every epoch."
)
print("=" * 70)