import os
import numpy as np
from PIL import Image

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
    return train_indices, val_indices, test_indices





X, Y = get_features_dog("dataset")
X, Y, class_names = prepare_data(X, Y)

train_indices, val_indices, test_indices = split_data(X, Y)

X_train, Y_train = X[train_indices], Y[train_indices]
X_val, Y_val = X[val_indices], Y[val_indices]
X_test, Y_test = X[test_indices], Y[test_indices]


