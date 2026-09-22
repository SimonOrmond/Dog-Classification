import os
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "images", "Images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations", "Annotation")
USE_BBOX_CROP = True

# Artifacts written by organise.py after training, read back by predict.py.
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_resnet.pt")
NORMALISATION_STATS_PATH = os.path.join(MODELS_DIR, "normalisation_stats.pt")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.pt")


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


def load_dataset():
    """Runs the full deterministic load+split pipeline (same seed as training) and returns
    (X, Y, class_names, train_indices, val_indices, test_indices)."""
    np.random.seed(42)
    check_duplicates(DATA_DIR)
    X, Y = get_features_dog(DATA_DIR, annotation_folder=ANNOTATION_DIR if USE_BBOX_CROP else None)
    X, Y, class_names = prepare_data(X, Y)
    train_indices, val_indices, test_indices = split_data(X, Y)
    return X, Y, class_names, train_indices, val_indices, test_indices
