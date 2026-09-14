import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EMBEDDING_PATH = "data/processed/esm2_embeddings.npy"
METADATA_PATH = "data/processed/esm2_metadata.csv"
THRESHOLD_PATH = "models/tuned_thresholds.json"

TARGET_COLUMNS = [
    "nucleoplasm",
    "cytosol",
    "vesicles",
    "plasma_membrane",
    "mitochondria",
    "golgi_apparatus",
    "nucleoli",
    "endoplasmic_reticulum",
]

RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 40
PATIENCE = 5


class LocalizationNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_probabilities(model, X):
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def tune_thresholds(y_true, probabilities):
    """
    Find the best threshold for each class using
    validation-set F1 score only.
    """

    thresholds = []

    candidate_thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for class_index, class_name in enumerate(TARGET_COLUMNS):
        best_threshold = 0.5
        best_f1 = -1.0

        true_labels = y_true[:, class_index]
        class_probabilities = probabilities[:, class_index]

        for threshold in candidate_thresholds:
            predictions = (
                class_probabilities >= threshold
            ).astype(int)

            score = f1_score(
                true_labels,
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds.append(best_threshold)

        print(
            f"{class_name:25s} "
            f"threshold={best_threshold:.2f} "
            f"validation F1={best_f1:.3f}"
        )

    return np.array(thresholds)


def evaluate(y_true, probabilities, thresholds):
    predictions = (
        probabilities >= thresholds
    ).astype(int)

    micro_f1 = f1_score(
        y_true,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    return micro_f1, macro_f1, predictions


def main():
    set_seed(RANDOM_SEED)

    print("Loading ESM-2 embeddings...")

    X = np.load(EMBEDDING_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    y = metadata[TARGET_COLUMNS].values.astype(
        np.float32
    )

    print(f"Embedding matrix: {X.shape}")
    print(f"Label matrix: {y.shape}")

    indices = np.arange(len(X))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=RANDOM_SEED,
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Training proteins: {len(X_train)}")
    print(f"Validation proteins: {len(X_val)}")
    print(f"Testing proteins: {len(X_test)}")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    training_dataset = TensorDataset(
        X_train,
        y_train_tensor,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    positive_counts = y_train_tensor.sum(dim=0)
    negative_counts = len(y_train_tensor) - positive_counts

    pos_weight = negative_counts / positive_counts

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    model = LocalizationNetwork(
        input_size=X.shape[1],
        output_size=len(TARGET_COLUMNS),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\nTraining neural network...")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for batch_X, batch_y in training_loader:
            optimizer.zero_grad()

            logits = model(batch_X)

            loss = loss_function(
                logits,
                batch_y,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_train_loss = (
            total_loss / len(training_loader)
        )

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)

            val_loss = loss_function(
                val_logits,
                y_val_tensor,
            ).item()

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {average_train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_state = {
                key: value.clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_state)

    # -----------------------------------
    # Tune thresholds on VALIDATION data
    # -----------------------------------

    print("\nTuning class-specific thresholds...")

    val_probabilities = get_probabilities(
        model,
        X_val,
    )

    tuned_thresholds = tune_thresholds(
        y_val,
        val_probabilities,
    )

    threshold_dict = {
        class_name: round(float(threshold), 2)
        for class_name, threshold
        in zip(TARGET_COLUMNS, tuned_thresholds)
    }

    with open(THRESHOLD_PATH, "w") as file:
        json.dump(
            threshold_dict,
            file,
            indent=4,
        )

    print(
        f"\nSaved thresholds to: {THRESHOLD_PATH}"
    )

    # -----------------------------------
    # Final evaluation on TEST data
    # -----------------------------------

    test_probabilities = get_probabilities(
        model,
        X_test,
    )

    default_thresholds = np.full(
        len(TARGET_COLUMNS),
        0.5,
    )

    default_micro, default_macro, _ = evaluate(
        y_test,
        test_probabilities,
        default_thresholds,
    )

    tuned_micro, tuned_macro, tuned_predictions = evaluate(
        y_test,
        test_probabilities,
        tuned_thresholds,
    )

    print("\nThreshold Comparison")
    print("--------------------")

    print("\nDefault threshold = 0.50")
    print(f"Micro F1: {default_micro:.3f}")
    print(f"Macro F1: {default_macro:.3f}")

    print("\nValidation-tuned thresholds")
    print(f"Micro F1: {tuned_micro:.3f}")
    print(f"Macro F1: {tuned_macro:.3f}")

    print("\nPer-class tuned results:")

    print(
        classification_report(
            y_test,
            tuned_predictions,
            target_names=TARGET_COLUMNS,
            zero_division=0,
        )
    )

    print("\nFinal Model Comparison")
    print("----------------------")
    print("Amino-acid composition:")
    print("Micro F1: 0.320")
    print("Macro F1: 0.286")

    print("\nESM-2 + Logistic Regression:")
    print("Micro F1: 0.385")
    print("Macro F1: 0.331")

    print("\nESM-2 + Neural Network (0.50 threshold):")
    print(f"Micro F1: {default_micro:.3f}")
    print(f"Macro F1: {default_macro:.3f}")

    print("\nESM-2 + Neural Network (tuned thresholds):")
    print(f"Micro F1: {tuned_micro:.3f}")
    print(f"Macro F1: {tuned_macro:.3f}")


if __name__ == "__main__":
    main()