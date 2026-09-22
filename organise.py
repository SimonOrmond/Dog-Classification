import os
import time
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join("images", "Images")  
ANNOTATION_DIR = os.path.join("annotations", "Annotation")  
USE_BBOX_CROP = True 
NUM_EPOCHS = 120
BEST_MODEL_PATH = "best_model_resnet.pt"  

np.random.seed(42)
torch.manual_seed(42)

def check_duplicates(folder):
    seen = {}
    for dog_name in os.listdir(folder):
        dog_class = os.path.join(folder, dog_name)
        for img_name in os.listdir(dog_class):
            img_path = os.path.join(dog_class, img_name)
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in seen:
                raise ValueError(f"Duplicate images: {seen[file_hash]} and {img_path}")
            seen[file_hash] = img_path

def resize_and_crop(img, size):
    width, height = img.size
    scale = size / min(width, height)
    img = img.resize((round(width * scale), round(height * scale)))
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))

def crop_to_dog(img, annotation_path, margin=0.15):

    if not os.path.exists(annotation_path):
        return img

    root = ET.parse(annotation_path).getroot()

    best_box = None
    best_area = 0
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best_box = (xmin, ymin, xmax, ymax)

    if best_box is None:
        return img

    xmin, ymin, xmax, ymax = best_box
    pad_x = (xmax - xmin) * margin
    pad_y = (ymax - ymin) * margin

    width, height = img.size
    left = max(0, int(xmin - pad_x))
    top = max(0, int(ymin - pad_y))
    right = min(width, int(xmax + pad_x))
    bottom = min(height, int(ymax + pad_y))

    return img.crop((left, top, right, bottom))


def get_features_dog(folder, size=150, annotation_folder=None):
    images = []
    labels = []
    for dog_name in sorted(os.listdir(folder)):
        dog_class = os.path.join(folder, dog_name)
        breed = dog_name.split("-", 1)[1]#"n02088364-beagle"
        for img_name in os.listdir(dog_class):
            img_path = os.path.join(dog_class, img_name)
            img = Image.open(img_path).convert("RGB")
            if annotation_folder is not None:
                annotation_path = os.path.join(annotation_folder, dog_name, img_name[:-4])
                img = crop_to_dog(img, annotation_path)
            img = resize_and_crop(img, size)
            img = np.array(img)
            images.append(img)
            labels.append(breed)

    return np.array(images), np.array(labels)

def prepare_data(x, y):
    x = x.astype(np.float32) / 255.0
    class_names, y_encoded = np.unique(y, return_inverse=True)
    return x, y_encoded, class_names

def split_data(x, y):

    classes = np.unique(y)
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(0,len(classes)):
        indices = np.where(y == classes[i])[0]
        np.random.shuffle(indices)

        train_end = int(len(indices) * 0.70)
        val_end = int(len(indices) * 0.85)

        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


check_duplicates(DATA_DIR)
X, Y = get_features_dog(DATA_DIR, annotation_folder=ANNOTATION_DIR if USE_BBOX_CROP else None)
X, Y, class_names = prepare_data(X, Y)

counts = np.bincount(Y)
for name, count in zip(class_names, counts):
    print(f"{name}: {count} images")
if len(set(counts)) != 1:
    raise ValueError("Breeds have different numbers of images")

train_indices, val_indices, test_indices = split_data(X, Y)

X_train, Y_train = X[train_indices], Y[train_indices]
X_val, Y_val = X[val_indices], Y[val_indices]
X_test, Y_test = X[test_indices], Y[test_indices]


X_train = np.transpose(X_train, (0, 3, 1, 2))
X_val = np.transpose(X_val, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))

X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train, dtype=torch.long)

X_val = torch.tensor(X_val)
Y_val = torch.tensor(Y_val, dtype=torch.long)

X_test = torch.tensor(X_test)
Y_test = torch.tensor(Y_test, dtype=torch.long)


channel_mean = X_train.mean(dim=(0, 2, 3)).view(3, 1, 1)
channel_std = X_train.std(dim=(0, 2, 3)).view(3, 1, 1)

torch.save({"mean": channel_mean, "std": channel_std}, "normalisation_stats.pt")

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
    """A dog facing left is the same breed as one facing right."""
    if torch.rand(1).item() < p:
        image = image.flip(2)  
    return image

def random_crop(image, padding=12):
    """Pads the edges by mirroring, then cuts a random 150x150 window, so the dog moves around."""
    _, height, width = image.shape
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    top = torch.randint(0, 2 * padding + 1, (1,)).item()
    left = torch.randint(0, 2 * padding + 1, (1,)).item()
    return padded[:, top:top + height, left:left + width]

def random_brightness(image, max_change=0.2):
    """Makes the photo up to 20% darker or brighter, like different lighting."""
    factor = 1 + (torch.rand(1).item() * 2 - 1) * max_change
    return (image * factor).clamp(0, 1)

def normalise(image):
    """Shifts and scales each colour channel to mean ~0, std ~1 using the training set's stats."""
    return (image - channel_mean) / channel_std

def train_augmentation(image):
    image = random_horizontal_flip(image)
    image = random_crop(image)
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


class ResidualBlock(nn.Module):
    """Two 3x3 convs, then adds the block's input back onto the output (the "shortcut").

    The shortcut means the block only has to learn what to change about its input,
    and gives gradients a direct path back to early layers, so a deeper network still trains.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # stride=2 halves height and width, doing the job max pooling did before
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # The input can only be added to the output if their shapes match. When the block
        # changes the channel count or shrinks the image, a 1x1 conv reshapes the input to fit.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # the residual connection
        return self.relu(out)


class DogCNN(nn.Module):
    """A small ResNet: a stem conv, four residual blocks (8 convs), global average pooling, one fully connected layer."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block1 = ResidualBlock(32, 32, stride=1)
        self.block2 = ResidualBlock(32, 64, stride=2)
        self.block3 = ResidualBlock(64, 128, stride=2)
        self.block4 = ResidualBlock(128, 256, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)     # (batch, 32, 75, 75)
        x = self.block1(x)   # (batch, 32, 75, 75)
        x = self.block2(x)   # (batch, 64, 38, 38)
        x = self.block3(x)   # (batch, 128, 19, 19)
        x = self.block4(x)   # (batch, 256, 10, 10)

        x = self.global_pool(x)
        x = self.flatten(x)  # (batch, 256)
        x = self.fc(x)       # (batch, 10)
        return x

model = DogCNN()
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