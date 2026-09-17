import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
def get_features_dog(folder,size=(150, 150)):
    images = []
    labels = []
    for dog_name in os.listdir(folder):
        dog_class = os.path.join(folder, dog_name)
        for img_name in os.listdir(dog_class):
            img_path = os.path.join(dog_class, img_name)
            img = Image.open(img_path)
            img = img.resize(size)
            img = np.array(img)
            images.append(img)
            labels.append(dog_name)
            
    
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


X, Y = get_features_dog("dataset")
X, Y, class_names = prepare_data(X, Y)

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


for images, labels in train_loader:

    optimizer.zero_grad()

    outputs = model(images)

    loss = loss_function(outputs, labels)

    loss.backward()

    optimizer.step()

    #print(loss.item())



model.eval()

total_loss = 0
correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        loss = loss_function(outputs, labels)
        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

average_loss = total_loss / len(val_loader)
accuracy = correct / total

print("Validation loss:", average_loss)
print("Validation accuracy:", accuracy * 100, "%")