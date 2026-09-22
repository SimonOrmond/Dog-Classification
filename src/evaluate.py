import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import DogCNN
from data import BEST_MODEL_PATH, NORMALISATION_STATS_PATH, CLASS_NAMES_PATH, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(num_classes):
    model = DogCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def main():
    class_names = torch.load(CLASS_NAMES_PATH, weights_only=False)
    stats = torch.load(NORMALISATION_STATS_PATH, map_location=device, weights_only=True)
    mean, std = stats["mean"], stats["std"]
    model = load_model(num_classes=len(class_names))

    X, Y, loaded_class_names, _, _, test_indices = load_dataset()
    assert list(loaded_class_names) == list(class_names), (
        "class_names.pt doesn't match the dataset on disk - retrain with organise.py first"
    )

    X_test = np.transpose(X[test_indices], (0, 3, 1, 2))
    Y_test = Y[test_indices]

    X_test = torch.tensor(X_test).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=32)

    correct = 0
    total = 0
    correct_per_class = np.zeros(len(class_names), dtype=int)
    total_per_class = np.zeros(len(class_names), dtype=int)

    with torch.no_grad():
        for images, labels in loader:
            normalised = (images - mean) / std
            predictions = torch.argmax(model(normalised), dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for label, prediction in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                total_per_class[label] += 1
                if label == prediction:
                    correct_per_class[label] += 1

    print(f"Test set accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    print()
    for name, right, seen in zip(class_names, correct_per_class, total_per_class):
        print(f"  {name}: {right}/{seen} ({100 * right / seen:.1f}%)")


if __name__ == "__main__":
    main()
