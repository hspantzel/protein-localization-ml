import pandas as pd
import matplotlib.pyplot as plt


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

DISPLAY_NAMES = [
    "Nucleoplasm",
    "Cytosol",
    "Vesicles",
    "Plasma membrane",
    "Mitochondria",
    "Golgi apparatus",
    "Nucleoli",
    "Endoplasmic reticulum",
]

# Results from our baseline experiment
F1_SCORES = [
    0.56,
    0.40,
    0.29,
    0.30,
    0.23,
    0.19,
    0.18,
    0.14,
]


def plot_class_distribution(df):
    counts = [df[column].sum() for column in TARGET_COLUMNS]

    plt.figure(figsize=(10, 6))
    plt.barh(DISPLAY_NAMES, counts)
    plt.xlabel("Number of proteins")
    plt.ylabel("Cellular location")
    plt.title("Protein Localization Class Distribution")
    plt.tight_layout()

    plt.savefig(
        "figures/class_distribution.png",
        dpi=300,
    )

    plt.close()


def plot_f1_scores():
    plt.figure(figsize=(10, 6))
    plt.barh(DISPLAY_NAMES, F1_SCORES)
    plt.xlabel("F1 score")
    plt.ylabel("Cellular location")
    plt.title("Baseline Model Performance")
    plt.xlim(0, 1)
    plt.tight_layout()

    plt.savefig(
        "figures/baseline_f1_scores.png",
        dpi=300,
    )

    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    plot_class_distribution(df)
    plot_f1_scores()

    print("Saved:")
    print("- figures/class_distribution.png")
    print("- figures/baseline_f1_scores.png")


if __name__ == "__main__":
    main()