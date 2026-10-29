from datasets import load_dataset
import glob
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

CLASSES = ["bicycle", "bus", "car", "crosswalk", "hydrant"]
SIZE = 64


def load_local_split(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"Не найдены файлы по шаблону: {pattern}")
    return sorted(files)


def prepare_data(split):
    images = []
    labels = []

    for item in split:
        image = item["image"].convert("RGB").resize((SIZE, SIZE))
        images.append(np.array(image, dtype=np.uint8))
        labels.append(item["labels"])

    images = np.array(images, dtype=np.uint8).transpose(0, 3, 1, 2)
    labels = np.array(labels, dtype=np.float32)

    return torch.from_numpy(images), torch.from_numpy(labels)


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, count = 0, 0, 0

    for images, labels in loader:
        images = images.to(device).float() / 255.0
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = (torch.sigmoid(out) >= 0.5).float()
        correct += (pred == labels).sum().item()
        count += labels.numel()

    return total_loss / len(loader.dataset), correct / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, count = 0, 0, 0
    all_images, all_labels, all_probs = [], [], []

    for images, labels in loader:
        inputs = images.to(device).float() / 255.0
        labels_device = labels.to(device)
        out = model(inputs)
        loss = loss_fn(out, labels_device)

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(out)
        pred = (probs >= 0.5).float()
        correct += (pred == labels_device).sum().item()
        count += labels.numel()

        all_images.append(images)
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (
        total_loss / len(loader.dataset),
        correct / count,
        torch.cat(all_images),
        torch.cat(all_labels),
        torch.cat(all_probs)
    )


def show_examples(images, labels, probs, amount=12):
    amount = min(amount, len(images))
    cols = 4
    rows = (amount + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(amount):
        image = images[i].permute(1, 2, 0).numpy()
        true_classes = [CLASSES[j] for j, v in enumerate(labels[i]) if v == 1]
        pred_classes = [CLASSES[j] for j, v in enumerate(probs[i]) if v >= 0.5]

        class_probabilities = ', '.join(
            f"{CLASSES[j]}: {probs[i][j].item() * 100:.0f}%"
            for j in range(len(CLASSES))
        )

        axes[i].imshow(image)
        axes[i].set_title(
            f"True: {', '.join(true_classes) or 'none'}\n"
            f"Pred: {', '.join(pred_classes) or 'none'}\n"
            f"{class_probabilities}",
            fontsize=8
        )
        axes[i].axis("off")

    for i in range(amount, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Укажите путь к скачанным локально Parquet-файлам.
    data_dir = "data/hf_dataset"

    train_files = load_local_split(data_dir + "/**/train-*.parquet")
    val_files = load_local_split(data_dir + "/**/validation-*.parquet")
    test_files = load_local_split(data_dir + "/**/test-*.parquet")

    train_data = load_dataset("parquet", data_files=train_files, split="train")
    val_data = load_dataset("parquet", data_files=val_files, split="train")
    test_data = load_dataset("parquet", data_files=test_files, split="train")

    print("Подготовка изображений...")
    train_images, train_labels = prepare_data(train_data)
    val_images, val_labels = prepare_data(val_data)
    test_images, test_labels = prepare_data(test_data)

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASSES))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 10
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{epochs} "
            f"train_acc={train_acc:.3f}, train_loss={train_loss:.3f} "
            f"val_acc={val_acc:.3f}, val_loss={val_loss:.3f}"
        )

    test_loss, test_acc, images, labels, probs = evaluate(
        model, test_loader, loss_fn, device
    )
    print(f"Test accuracy (per-label): {test_acc:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    show_examples(images, labels, probs)
