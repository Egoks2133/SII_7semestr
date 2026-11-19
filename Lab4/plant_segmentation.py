import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter

from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp


DATASET_NAME = "Project-AgML/image_safari_plant_part_segmentation"

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.0001

NUM_CLASSES = 18

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MODEL_PATH = "model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

indices = np.arange(len(dataset))

if "file_name" in dataset.column_names:
    groups = np.array(dataset["file_name"])
else:
    groups = indices

split_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_RATIO,
    random_state=SEED
)

train_val_indices, test_indices = next(
    split_1.split(indices, groups=groups)
)

val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

split_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=val_fraction,
    random_state=SEED
)

train_indices_local, val_indices_local = next(
    split_2.split(
        train_val_indices,
        groups=groups[train_val_indices]
    )
)

train_indices = train_val_indices[train_indices_local]
val_indices = train_val_indices[val_indices_local]

train_data = dataset.select(train_indices.tolist())
val_data = dataset.select(val_indices.tolist())
test_data = dataset.select(test_indices.tolist())


class PlantSegmentationDataset(Dataset):

    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image = item["image"]
        mask = item["mask"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        image = image.convert("RGB")

        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.asarray(mask))

        if mask.mode != "L":
            mask = mask.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Размеры изображения и маски не совпадают: "
                f"{image.size} и {mask.size}"
            )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])

                image = TF.rotate(
                    image,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR
                )

                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST
                )

            image = self.color_jitter(image)

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=TF.InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=TF.InterpolationMode.NEAREST
        )

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask = np.array(mask, dtype=np.int64)

        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(
                f"Обнаружено недопустимое значение маски: "
                f"min={mask.min()}, max={mask.max()}. "
                f"Ожидаются значения от 0 до {NUM_CLASSES - 1}."
            )

        mask = torch.from_numpy(mask).long()

        return image, mask


train_dataset = PlantSegmentationDataset(
    train_data,
    augment=True
)

val_dataset = PlantSegmentationDataset(
    val_data,
    augment=False
)

test_dataset = PlantSegmentationDataset(
    test_data,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
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


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
)

model = model.to(DEVICE)


class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        probabilities = torch.softmax(
            predictions,
            dim=1
        )

        targets_one_hot = torch.nn.functional.one_hot(
            targets,
            num_classes=NUM_CLASSES
        )

        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        ).float()

        probabilities = probabilities[:, 1:, :, :]
        targets_one_hot = targets_one_hot[:, 1:, :, :]

        intersection = (
            probabilities * targets_one_hot
        ).sum(dim=(0, 2, 3))

        union = (
            probabilities + targets_one_hot
        ).sum(dim=(0, 2, 3))

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1.0 - dice.mean()


cross_entropy = nn.CrossEntropyLoss()
dice_loss = DiceLoss()


def calculate_loss(predictions, targets):
    ce = cross_entropy(
        predictions,
        targets
    )

    dice = dice_loss(
        predictions,
        targets
    )

    return ce + dice


optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=3,
    factor=0.5
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


def calculate_metrics(predictions, targets):
    predictions = torch.argmax(
        predictions,
        dim=1
    )

    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    dice_intersection = np.zeros(NUM_CLASSES)
    dice_union = np.zeros(NUM_CLASSES)

    correct = 0
    total = 0

    for pred, target in zip(predictions, targets):
        correct += np.sum(pred == target)
        total += target.size

        for class_id in range(NUM_CLASSES):
            pred_mask = pred == class_id
            target_mask = target == class_id

            inter = np.logical_and(
                pred_mask,
                target_mask
            ).sum()

            pred_area = pred_mask.sum()
            target_area = target_mask.sum()

            intersection[class_id] += inter

            union[class_id] += (
                pred_area + target_area - inter
            )

            dice_intersection[class_id] += inter

            dice_union[class_id] += (
                pred_area + target_area
            )

    iou = intersection / (union + 1e-7)

    dice = (
        2 * dice_intersection
    ) / (
        dice_union + 1e-7
    )

    valid_classes = union[1:] > 0

    if valid_classes.any():
        mean_iou = np.mean(iou[1:][valid_classes])
        mean_dice = np.mean(dice[1:][valid_classes])
    else:
        mean_iou = 0.0
        mean_dice = 0.0

    pixel_accuracy = correct / total

    return pixel_accuracy, mean_iou, mean_dice


def train_one_epoch(model, loader):
    model.train()

    running_loss = 0.0

    for images, masks in tqdm(
        loader,
        desc="Обучение",
        leave=False
    ):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda"
        ):
            predictions = model(images)
            loss = calculate_loss(
                predictions,
                masks
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()

    running_loss = 0.0

    all_accuracy = []
    all_iou = []
    all_dice = []

    with torch.no_grad():
        for images, masks in tqdm(
            loader,
            desc="Оценка",
            leave=False
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda"
            ):
                predictions = model(images)
                loss = calculate_loss(
                    predictions,
                    masks
                )

            running_loss += loss.item()

            accuracy, iou, dice = calculate_metrics(
                predictions,
                masks
            )

            all_accuracy.append(accuracy)
            all_iou.append(iou)
            all_dice.append(dice)

    return (
        running_loss / len(loader),
        np.mean(all_accuracy),
        np.mean(all_iou),
        np.mean(all_dice)
    )


def train_model():
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_iou": [],
        "val_dice": []
    }

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader
        )

        (
            val_loss,
            val_accuracy,
            val_iou,
            val_dice
        ) = evaluate(
            model,
            val_loader
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "image_size": IMAGE_SIZE
                },
                MODEL_PATH
            )

    return history


def plot_history(history):
    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Изменение функции потерь")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        history["val_iou"],
        label="Mean IoU"
    )

    plt.plot(
        epochs,
        history["val_dice"],
        label="Mean Dice"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Pixel Accuracy"
    )

    plt.xlabel("Эпоха")
    plt.ylabel("Значение метрики")
    plt.title("Метрики качества")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, dataset, count=5):
    model.eval()

    count = min(
        count,
        len(dataset)
    )

    indices = random.sample(
        range(len(dataset)),
        count
    )

    with torch.no_grad():
        for index in indices:
            image, mask = dataset[index]

            input_tensor = image.unsqueeze(0).to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=DEVICE.type == "cuda"
            ):
                prediction = model(input_tensor)

            prediction = torch.argmax(
                prediction,
                dim=1
            ).squeeze(0).cpu().numpy()

            image_np = image.cpu().numpy()

            mean = np.array(
                [0.485, 0.456, 0.406]
            )

            std = np.array(
                [0.229, 0.224, 0.225]
            )

            image_np = (
                image_np.transpose(1, 2, 0) * std + mean
            )

            image_np = np.clip(
                image_np,
                0,
                1
            )

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title("Исходное изображение")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(mask.numpy(), cmap="tab20")
            plt.title("Настоящая маска")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap="tab20")
            plt.title("Предсказанная маска")
            plt.axis("off")

            plt.tight_layout()
            plt.show()


def predict_image(image_path):
    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    image = TF.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=TF.InterpolationMode.BILINEAR
    )

    image = TF.to_tensor(image)

    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE.type,
            enabled=DEVICE.type == "cuda"
        ):
            prediction = model(image)

        prediction = torch.argmax(
            prediction,
            dim=1
        ).squeeze(0).cpu().numpy()

    prediction_image = Image.fromarray(
        prediction.astype(np.uint8)
    )

    prediction_image = prediction_image.resize(
        original_image.size,
        Image.Resampling.NEAREST
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Исходное изображение")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(
        prediction_image,
        cmap="tab20",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )
    plt.title("Результат сегментации")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return prediction_image


if __name__ == "__main__":
    history = train_model()

    plot_history(history)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_loss,
        test_accuracy,
        test_iou,
        test_dice
    ) = evaluate(
        model,
        test_loader
    )

    results = {
        "test_loss": float(test_loss),
        "pixel_accuracy": float(test_accuracy),
        "mean_iou": float(test_iou),
        "mean_dice": float(test_dice)
    }

    print("\nИтоговые результаты:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Pixel Accuracy: {test_accuracy:.4f}")
    print(f"Mean IoU: {test_iou:.4f}")
    print(f"Mean Dice: {test_dice:.4f}")

    visualize_predictions(
        model,
        test_dataset,
        count=5
    )
