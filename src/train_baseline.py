import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from features import amino_acid_composition


DATA_PATH = "data/processed/protein_dataset.csv"

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
    df = pd.read_csv(DATA_PATH)

    print(f"Number of proteins: {len(df)}")

    # Convert each protein sequence into 20 amino-acid composition features
    feature_rows = df["sequence"].apply(amino_acid_composition)

    X = pd.DataFrame(feature_rows.tolist())
    y = df[TARGET_COLUMNS]

    print(f"Number of features: {X.shape[1]}")

    # 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training proteins: {len(X_train)}")
    print(f"Testing proteins: {len(X_test)}")

    # Train one logistic regression classifier per location
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
    )

    print("\nTraining model...")
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

    print("\nBaseline Results")
    print("----------------")
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


if __name__ == "__main__":
    main()