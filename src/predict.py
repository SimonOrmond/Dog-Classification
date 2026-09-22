import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from model import DogCNN
from data import (
    BEST_MODEL_PATH,
    NORMALISATION_STATS_PATH,
    CLASS_NAMES_PATH,
    resize_and_crop,
    load_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(num_classes):
    model = DogCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def load_image_from_path(path):
    """Preprocesses an arbitrary photo the same way training images were (minus the bounding-box
    crop, since a new photo has no annotation to crop to)."""
    img = Image.open(path).convert("RGB")
    img = resize_and_crop(img, 150)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1)


def predict(image, model, mean, std):
    """image: (3, 150, 150) tensor with values in [0, 1]. Returns (predicted_index, probabilities)."""
    normalised = (image.to(device) - mean) / std
    with torch.no_grad():
        logits = model(normalised.unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    predicted_index = torch.argmax(probs).item()
    return predicted_index, probs


def show_prediction(image, class_names, predicted_index, probs, actual_label=None):
    predicted_name = class_names[predicted_index]
    confidence = probs[predicted_index].item() * 100
    title = f"Predicted: {predicted_name} ({confidence:.1f}%)"
    if actual_label is not None:
        result = "correct" if actual_label == predicted_name else "WRONG"
        title += f"\nActual: {actual_label}  [{result}]"

    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


def main():
    class_names = torch.load(CLASS_NAMES_PATH, weights_only=False)
    stats = torch.load(NORMALISATION_STATS_PATH, map_location=device, weights_only=True)
    mean, std = stats["mean"], stats["std"]
    model = load_model(num_classes=len(class_names))

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Predicting {image_path}...")
        image = load_image_from_path(image_path)
        predicted_index, probs = predict(image, model, mean, std)
        show_prediction(image, class_names, predicted_index, probs)
        return

    print("No image path given, picking a random image from the test set instead.")
    X, Y, loaded_class_names, _, _, test_indices = load_dataset()
    assert list(loaded_class_names) == list(class_names), (
        "class_names.pt doesn't match the dataset on disk - retrain with organise.py first"
    )

    index = random.choice(list(test_indices))
    image = torch.tensor(X[index]).permute(2, 0, 1)
    actual_label = class_names[Y[index]]

    predicted_index, probs = predict(image, model, mean, std)
    show_prediction(image, class_names, predicted_index, probs, actual_label=actual_label)


if __name__ == "__main__":
    main()
