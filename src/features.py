"""Feature extraction utilities for protein sequences."""

from collections import Counter


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def amino_acid_composition(sequence: str) -> dict[str, float]:
    """Calculate the fraction of each amino acid in a protein sequence."""

    sequence = sequence.strip().upper()
    counts = Counter(sequence)

    valid_length = sum(counts[aa] for aa in AMINO_ACIDS)

    if valid_length == 0:
        raise ValueError("Sequence contains no standard amino acids.")

    return {
        f"aa_{aa}": counts[aa] / valid_length
        for aa in AMINO_ACIDS
    }


if __name__ == "__main__":
    example_sequence = "MKWVTFISLLFLFSSAYSRGVFRR"

    features = amino_acid_composition(example_sequence)

    for amino_acid, value in features.items():
        print(f"{amino_acid}: {value:.3f}")