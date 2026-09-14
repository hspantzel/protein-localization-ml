"""Feature extraction utilities for protein sequences."""

from collections import Counter


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def amino_acid_composition(sequence: str) -> dict[str, float]:
    """Calculate the fraction of each amino acid in a protein sequence."""

    sequence = sequence.strip().upper()

    if not sequence:
        raise ValueError("Sequence is empty.")

    invalid_amino_acids = set(sequence) - set(AMINO_ACIDS)

    if invalid_amino_acids:
        invalid = ", ".join(sorted(invalid_amino_acids))
        raise ValueError(
            f"Sequence contains invalid amino acids: {invalid}"
        )

    counts = Counter(sequence)

    return {
        f"aa_{aa}": counts[aa] / len(sequence)
        for aa in AMINO_ACIDS
    }


if __name__ == "__main__":
    example_sequence = "MKWVTFISLLFLFSSAYSRGVFRR"

    features = amino_acid_composition(example_sequence)

    for amino_acid, value in features.items():
        print(f"{amino_acid}: {value:.3f}")