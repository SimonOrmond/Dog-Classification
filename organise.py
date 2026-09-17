import os
import hashlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join("images", "Images")  # Stanford Dogs, one folder per breed
NUM_EPOCHS = 40
BEST_MODEL_PATH = "best_model.pt"  # weights from the epoch with the lowest validation loss

np.random.seed(42)
torch.manual_seed(42)

def check_duplicates(folder):
    """Stops the program if any two image files are byte-for-byte identical."""
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
    """Shrinks the shorter side to `size`, then cuts a centre square, so dogs aren't stretched."""
    width, height = img.size
    scale = size / min(width, height)
    img = img.resize((round(width * scale), round(height * scale)))
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))

def get_features_dog(folder, size=150):
    images = []
    labels = []
    for dog_name in sorted(os.listdir(folder)):
        dog_class = os.path.join(folder, dog_name)
        # Stanford folder names look like "n02088364-beagle", keep only the breed
        breed = dog_name.split("-", 1)[1]
        for img_name in os.listdir(dog_class):
            img_path = os.path.join(dog_class, img_name)
            img = Image.open(img_path).convert("RGB")
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
X, Y = get_features_dog(DATA_DIR)
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


# Augmentation: small random changes that don't change the breed.
# Each image is a (3, 150, 150) tensor with values 0-1. None of these edit
# the stored image, so every epoch gets a fresh random version of each photo.

def random_horizontal_flip(image, p=0.5):
    """A dog facing left is the same breed as one facing right."""
    if torch.rand(1).item() < p:
        image = image.flip(2)  # dim 2 is width
    return image

def random_crop(image, padding=12):
    """Pads the edges by mirroring, then cuts a random 150x150 window, so the dog moves around."""
    _, height, width = image.shape
    # reflect padding needs a batch dimension, so add one then remove it
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    top = torch.randint(0, 2 * padding + 1, (1,)).item()
    left = torch.randint(0, 2 * padding + 1, (1,)).item()
    return padded[:, top:top + height, left:left + width]

def random_brightness(image, max_change=0.2):
    """Makes the photo up to 20% darker or brighter, like different lighting."""
    factor = 1 + (torch.rand(1).item() * 2 - 1) * max_change
    return (image * factor).clamp(0, 1)

def train_augmentation(image):
    image = random_horizontal_flip(image)
    image = random_crop(image)
    image = random_brightness(image)
    return image


# Only the training set is augmented, validation and test stay fixed
train_dataset = DogDataset(X_train, Y_train, transform=train_augmentation)
val_dataset = DogDataset(X_val, Y_val)
test_dataset = DogDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

images, labels = next(iter(train_loader))


class DogCNN(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Pool, four times, then global average pooling and one fully connected layer."""

    # __init__ runs once, when you write DogCNN(). It creates the layers and their weights.
    def __init__(self, num_classes=10):
        # Runs nn.Module's own setup first, which lets PyTorch track every
        # layer assigned to self.something below (so model.parameters() finds them)
        super().__init__()

        # bias=False: BatchNorm straight after subtracts the channel's average,
        # which cancels out any constant a bias would add. Its own shift (beta) does that job instead.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)

        # BatchNorm2d(C) normalises each of C channels separately, so C must equal
        # the out_channels of the conv before it. Each one learns a scale (gamma) and a
        # shift (beta) per channel, and keeps a running mean/variance for use in eval mode.
        # Unlike pool and relu, these can't be shared: each holds different learned numbers.
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # Averages each of the 256 channels over the whole 9x9 map, giving 256 numbers
        # no matter where in the image a feature appeared
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # 150 -> pool 75 -> pool 37 -> pool 18 -> pool 9 -> global pool 1
        self.fc = nn.Linear(256, num_classes)

    # forward runs every time you call model(images). x starts as (batch, 3, 150, 150).
    # BatchNorm goes between conv and relu, so the values relu sees are centred on 0
    # and roughly half of them pass through.
    def forward(self, x):
        x = self.conv1(x)   # (batch, 32, 150, 150)
        x = self.bn1(x)     # same shape, each channel rescaled to mean ~0, spread ~1
        x = self.relu(x)
        x = self.pool(x)    # (batch, 32, 75, 75)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)    # (batch, 64, 37, 37)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)    # (batch, 128, 18, 18)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)    # (batch, 256, 9, 9)

        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = DogCNN()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lowers the learning rate along half a cosine curve, from 0.001 at the start to ~0 by
# the last epoch: big steps early, small careful steps late. It must be created after the
# optimizer because it works by editing the optimizer's learning rate directly.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


def train_one_epoch(model, loader):
    # Training mode: BatchNorm normalises with the current batch's mean/variance
    # and updates its running averages
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

        # loss is an average over the batch, so weight it by batch size
        total_loss += loss.item() * labels.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader):
    # Eval mode: BatchNorm uses the running averages saved during training instead,
    # so an image's prediction doesn't depend on which other images share its batch
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


# Kept so the loss curves can be plotted afterwards
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

# float("inf") is infinity, so the first epoch always counts as an improvement
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    # Read the learning rate used for this epoch before the scheduler changes it.
    # param_groups is a list because an optimizer can hold groups with different
    # learning rates; this one has a single group, so it's [0].
    current_lr = optimizer.param_groups[0]["lr"]

    # Once per epoch, after all of this epoch's optimizer.step() calls
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    # Judged on val loss rather than accuracy: it's smoother, and it separates epochs
    # with the same accuracy but different confidence
    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        # state_dict() holds every learned weight plus BatchNorm's running mean/variance.
        # torch.save writes a snapshot to disk, so later epochs can't overwrite it.
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    # The * marks epochs that set a new best and were saved
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  lr {current_lr:.6f}  "
          f"train loss {train_loss:.3f} acc {train_acc * 100:.1f}%  |  "
          f"val loss {val_loss:.3f} acc {val_acc * 100:.1f}%"
          f"{'  *' if improved else ''}")

# Swap the last epoch's weights for the best epoch's.
# weights_only=True only loads tensors, not arbitrary Python objects: .pt files use
# pickle, which could run code when loaded, so it's a good habit.
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

    # Dotted line at the epoch whose weights were kept
    if best_epoch is not None:
        for ax in (loss_ax, acc_ax):
            ax.axvline(best_epoch, color="green", linestyle=":", label=f"Best (epoch {best_epoch})")
        loss_ax.legend()
    acc_ax.legend()

    fig.tight_layout()
    plt.show()


plot_history(history, best_epoch)