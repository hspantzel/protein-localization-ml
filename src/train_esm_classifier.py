import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EMBEDDING_PATH = "data/processed/esm2_embeddings.npy"
METADATA_PATH = "data/processed/esm2_metadata.csv"

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


def main():
    print("Loading ESM-2 embeddings...")

    X = np.load(EMBEDDING_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    y = metadata[TARGET_COLUMNS]

    print(f"Embedding matrix: {X.shape}")
    print(f"Label matrix: {y.shape}")

    # Same random seed and 80/20 split as our baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training proteins: {len(X_train)}")
    print(f"Testing proteins: {len(X_test)}")

    # Scale the ESM features, then train one classifier
    # for each cellular location.
    model = OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        )
    )

    print("\nTraining classifier on ESM-2 embeddings...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    micro_f1 = f1_score(
        y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\nESM-2 Results")
    print("-------------")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    print("\nPer-class results:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=TARGET_COLUMNS,
            zero_division=0,
        )
    )

    print("\nComparison")
    print("----------")
    print("Amino-acid composition baseline:")
    print("Micro F1: 0.320")
    print("Macro F1: 0.286")

    print("\nESM-2 embeddings:")
    print(f"Micro F1: {micro_f1:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")


if __name__ == "__main__":
    main()