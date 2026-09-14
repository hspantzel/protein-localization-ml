import matplotlib.pyplot as plt
import numpy as np


LOCATIONS = [
    "Nucleoplasm",
    "Cytosol",
    "Vesicles",
    "Plasma membrane",
    "Mitochondria",
    "Golgi apparatus",
    "Nucleoli",
    "Endoplasmic reticulum",
]

BASELINE_F1 = [
    0.56,
    0.40,
    0.29,
    0.30,
    0.23,
    0.19,
    0.18,
    0.14,
]

ESM_F1 = [
    0.64,
    0.42,
    0.32,
    0.40,
    0.30,
    0.19,
    0.17,
    0.21,
]


def plot_overall_comparison():
    model_names = [
        "Amino-acid\ncomposition",
        "ESM-2",
    ]

    micro_scores = [0.320, 0.385]
    macro_scores = [0.286, 0.331]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(
        x - width / 2,
        micro_scores,
        width,
        label="Micro F1",
    )

    plt.bar(
        x + width / 2,
        macro_scores,
        width,
        label="Macro F1",
    )

    plt.xticks(x, model_names)
    plt.ylabel("F1 score")
    plt.ylim(0, 0.5)
    plt.title("Baseline vs ESM-2 Protein Embeddings")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "figures/model_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_per_class_comparison():
    y = np.arange(len(LOCATIONS))
    height = 0.35

    plt.figure(figsize=(10, 7))

    plt.barh(
        y - height / 2,
        BASELINE_F1,
        height,
        label="Amino-acid composition",
    )

    plt.barh(
        y + height / 2,
        ESM_F1,
        height,
        label="ESM-2",
    )

    plt.yticks(y, LOCATIONS)
    plt.xlabel("F1 score")
    plt.xlim(0, 0.75)
    plt.title("Per-Class F1: Baseline vs ESM-2")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "figures/per_class_model_comparison.png",
        dpi=300,
    )

    plt.close()


def main():
    plot_overall_comparison()
    plot_per_class_comparison()

    print("Saved:")
    print("- figures/model_comparison.png")
    print("- figures/per_class_model_comparison.png")


if __name__ == "__main__":
    main()