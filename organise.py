import os
import hashlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join("images", "Images")  # Stanford Dogs, one folder per breed
NUM_EPOCHS = 15

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


train_dataset = DogDataset(X_train, Y_train)
val_dataset = DogDataset(X_val, Y_val)
test_dataset = DogDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

images, labels = next(iter(train_loader))


class DogCNN(nn.Module):
    """Conv -> ReLU -> Pool, twice, then one fully connected layer to 10 classes."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 150 -> conv(pad=1) 150 -> pool 75 -> conv(pad=1) 75 -> pool 37
        self.fc = nn.Linear(64 * 37 * 37, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x

model = DogCNN()

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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

        # loss is an average over the batch, so weight it by batch size
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


# Kept so the loss curves can be plotted afterwards
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
          f"train loss {train_loss:.3f} acc {train_acc * 100:.1f}%  |  "
          f"val loss {val_loss:.3f} acc {val_acc * 100:.1f}%")