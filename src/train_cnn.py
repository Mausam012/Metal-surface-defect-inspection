
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "dataset" / "processed"
    / "cnn_crops_manifest.csv"
)

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "silk_spot",
    "inclusion",
    "waist_folding",
    "rolled_pit"
]

NUM_CLASSES = len(CLASS_NAMES)

torch.manual_seed(42)
np.random.seed(42)

print("Training device:", DEVICE)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(MANIFEST)

df["cnn_label"] = df["cnn_label"].astype(int)

print("\nTotal crops:", len(df))
print("\nClass distribution:")
print(df["class_name"].value_counts())

# ==========================================
# 3. GROUP-WISE DATA SPLITTING
# ==========================================

# All crops from one original image must
# remain in the same split.

def create_group_splits(data):

    all_classes = set(range(NUM_CLASSES))

    for seed in range(1000):

        splitter1 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=seed
        )

        trainval_idx, test_idx = next(
            splitter1.split(
                data,
                groups=data["original_image"]
            )
        )

        trainval = data.iloc[trainval_idx]
        test = data.iloc[test_idx]

        if set(test["cnn_label"].unique()) != all_classes:
            continue

        if set(trainval["cnn_label"].unique()) != all_classes:
            continue

        splitter2 = GroupShuffleSplit(
            n_splits=1,
            test_size=0.1765,
            random_state=seed + 1000
        )

        train_idx, val_idx = next(
            splitter2.split(
                trainval,
                groups=trainval["original_image"]
            )
        )

        train = trainval.iloc[train_idx]
        val = trainval.iloc[val_idx]

        if (
            set(train["cnn_label"].unique()) == all_classes
            and set(val["cnn_label"].unique()) == all_classes
        ):

            return train.copy(), val.copy(), test.copy()

    raise ValueError(
        "Could not find group-wise splits containing "
        "all four classes. Review the rare class data."
    )


train_df, val_df, test_df = create_group_splits(df)

# Check for data leakage
train_groups = set(train_df["original_image"])
val_groups = set(val_df["original_image"])
test_groups = set(test_df["original_image"])

assert train_groups.isdisjoint(val_groups)
assert train_groups.isdisjoint(test_groups)
assert val_groups.isdisjoint(test_groups)

print("\n========== DATA SPLIT ==========")

for name, part in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    print(
        f"{name}: {len(part)} crops, "
        f"{part['original_image'].nunique()} original images"
    )

    print(part["class_name"].value_counts())

# Save split information
split_frames = []

for name, part in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:

    temp = part.copy()
    temp["split"] = name
    split_frames.append(temp)

split_manifest = pd.concat(
    split_frames,
    ignore_index=True
)

split_manifest.to_csv(
    ROOT / "dataset" / "processed"
    / "cnn_split_manifest.csv",
    index=False
)

# ==========================================
# 4. IMAGE TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ==========================================
# 5. PYTORCH DATASET
# ==========================================

class DefectDataset(Dataset):

    def __init__(self, dataframe, transform):

        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = ROOT / row["crop_path"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        label = int(row["cnn_label"])

        return image, label


train_dataset = DefectDataset(
    train_df, train_transform
)

val_dataset = DefectDataset(
    val_df, eval_transform
)

test_dataset = DefectDataset(
    test_df, eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# 6. BUILD CNN MODEL
# ==========================================

class DefectCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


model = DefectCNN().to(DEVICE)

print("\nCNN model created successfully.")

# ==========================================
# 7. LOSS AND OPTIMIZER
# ==========================================

class_counts = (
    train_df["cnn_label"]
    .value_counts()
    .reindex(range(NUM_CLASSES), fill_value=0)
    .values
)

if np.any(class_counts == 0):
    raise ValueError("A class is missing from training data.")

class_weights = (
    len(train_df) / (NUM_CLASSES * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# 8. TRAINING LOOP
# ==========================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
best_epoch = 0
patience = 8
wait = 0

print("\n========== CNN TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss = running_loss / train_total

    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss_total = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total

    val_accuracy = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best validation model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        wait = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "best_epoch": best_epoch
        }, MODEL_DIR / "cnn_four_class.pth")

    else:

        wait += 1

    if wait >= patience:

        print("Early stopping triggered.")

        break

print("\nBest validation epoch:", best_epoch)

# ==========================================
# 9. EVALUATE ON TEST SET
# ==========================================

checkpoint = torch.load(
    MODEL_DIR / "cnn_four_class.pth",
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n========== TEST RESULTS ==========")

print("Test accuracy:", round(test_accuracy, 4))

print("\nClassification report:")

print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

# ==========================================
# 10. SAVE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("CNN Confusion Matrix")
plt.colorbar()

plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45)
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png"
)

plt.close()

# ==========================================
# 11. SAVE TRAINING GRAPHS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig(RESULT_DIR / "training_loss.png")
plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(RESULT_DIR / "accuracy.png")
plt.close()

print("\nModel saved:", MODEL_DIR / "cnn_four_class.pth")
print("Results saved:", RESULT_DIR)

print("\n========== CNN TRAINING COMPLETED ==========")
