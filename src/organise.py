import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import DogCNN
from data import BEST_MODEL_PATH, NORMALISATION_STATS_PATH, CLASS_NAMES_PATH, load_dataset

NUM_EPOCHS = 120

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X, Y, class_names, train_indices, val_indices, test_indices = load_dataset()
torch.save(list(class_names), CLASS_NAMES_PATH)

counts = np.bincount(Y)
for name, count in zip(class_names, counts):
    print(f"{name}: {count} images")
if len(set(counts)) != 1:
    raise ValueError("Breeds have different numbers of images")

X_train, Y_train = X[train_indices], Y[train_indices]
X_val, Y_val = X[val_indices], Y[val_indices]
X_test, Y_test = X[test_indices], Y[test_indices]


X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val = np.transpose(X_val, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

X_train = torch.tensor(X_train).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)

X_val = torch.tensor(X_val).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)

X_test = torch.tensor(X_test).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)


channel_mean = X_train.mean(dim=(0, 2, 3)).view(3, 1, 1)
channel_std = X_train.std(dim=(0, 2, 3)).view(3, 1, 1)

torch.save({"mean": channel_mean, "std": channel_std}, NORMALISATION_STATS_PATH)

class DogDataset(Dataset):
    """Serves one (image, label) pair at a time so a DataLoader can batch them."""

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image = self.x[index]
        label = self.y[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def random_horizontal_flip(image, p=0.5):
    if torch.rand(1).item() < p:
        image = image.flip(2)
    return image

def random_crop(image, padding=12):
    _, height, width = image.shape
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    top = torch.randint(0, 2 * padding + 1, (1,)).item()
    left = torch.randint(0, 2 * padding + 1, (1,)).item()
    return padded[:, top:top + height, left:left + width]

def random_resized_crop(image, size=150, scale=(0.8, 1.0)):

    _, height, width = image.shape
    area = height * width
    target_area = area * (scale[0] + torch.rand(1).item() * (scale[1] - scale[0]))
    crop_size = min(int(round(target_area ** 0.5)), height, width)
    top = torch.randint(0, height - crop_size + 1, (1,)).item()
    left = torch.randint(0, width - crop_size + 1, (1,)).item()
    cropped = image[:, top:top + crop_size, left:left + crop_size]
    return F.interpolate(cropped.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)

def random_rotation(image, max_degrees=15):

    angle = torch.empty(1).uniform_(-max_degrees, max_degrees).item()
    theta = torch.tensor(angle * torch.pi / 180)
    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
    ], dtype=image.dtype, device=image.device).unsqueeze(0)
    grid = F.affine_grid(rotation_matrix, [1, *image.shape], align_corners=False)
    rotated = F.grid_sample(image.unsqueeze(0), grid, mode="bicubic", align_corners=False, padding_mode="reflection")
    return rotated.squeeze(0)

def random_saturation(image, max_change=0.3):

    factor = 1 + (torch.rand(1).item() * 2 - 1) * max_change
    luminance_weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device).view(3, 1, 1)
    grey = (image * luminance_weights).sum(dim=0, keepdim=True)
    return (grey + (image - grey) * factor).clamp(0, 1)

def random_contrast(image, max_change=0.2):

    factor = 1 + (torch.rand(1).item() * 2 - 1) * max_change
    mean = image.mean()
    return ((image - mean) * factor + mean).clamp(0, 1)

def random_brightness(image, max_change=0.2):

    factor = 1 + (torch.rand(1).item() * 2 - 1) * max_change
    return (image * factor).clamp(0, 1)

def normalise(image):

    return (image - channel_mean) / channel_std

def train_augmentation(image):
    image = random_horizontal_flip(image)
    image = random_crop(image)
    image = random_rotation(image)
    image = random_saturation(image)  # colour jitter: hue is left alone since coat colour is a real breed cue
    image = random_contrast(image)
    image = random_brightness(image)  # still on 0-1 values here, so its clamp(0, 1) is safe
    image = normalise(image)          # must come last: normalised values go negative
    return image

train_dataset = DogDataset(X_train, Y_train, transform=train_augmentation)
val_dataset = DogDataset(X_val, Y_val, transform=normalise)
test_dataset = DogDataset(X_test, Y_test, transform=normalise)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

images, labels = next(iter(train_loader))

model = DogCNN(num_classes=len(class_names)).to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


def train_one_epoch(model, loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

best_val_loss = float("inf")
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    current_lr = optimizer.param_groups[0]["lr"]

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)


    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  lr {current_lr:.6f}  "
          f"train loss {train_loss:.3f} acc {train_acc * 100:.1f}%  |  "
          f"val loss {val_loss:.3f} acc {val_acc * 100:.1f}%  "
          f"({time.time() - epoch_start:.0f}s)"
          f"{'  *' if improved else ''}")

model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
print(f"Loaded best model from epoch {best_epoch} (val loss {best_val_loss:.3f})")


def plot_history(history, best_epoch=None):
    """Draws train vs validation loss and accuracy side by side."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_ax.plot(epochs, history["train_loss"], label="Train")
    loss_ax.plot(epochs, history["val_loss"], label="Validation")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.legend()

    acc_ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    acc_ax.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    acc_ax.axhline(100 / len(class_names), color="grey", linestyle="--", label="Random guess")
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy (%)")
    acc_ax.set_ylim(0, 100)

    if best_epoch is not None:
        for ax in (loss_ax, acc_ax):
            ax.axvline(best_epoch, color="green", linestyle=":", label=f"Best (epoch {best_epoch})")
        loss_ax.legend()
    acc_ax.legend()

    fig.tight_layout()
    plt.show()


plot_history(history, best_epoch)
