import pytest

from src.features import amino_acid_composition


def test_composition_sums_to_one():
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    features = amino_acid_composition(sequence)

    assert abs(sum(features.values()) - 1.0) < 1e-9


def test_lowercase_sequence():
    sequence = "acdefghiklmnpqrstvwy"

    features = amino_acid_composition(sequence)

    assert abs(sum(features.values()) - 1.0) < 1e-9
    assert features["aa_A"] == pytest.approx(0.05)


def test_repeated_amino_acid():
    sequence = "AAAA"

    features = amino_acid_composition(sequence)

    assert features["aa_A"] == 1.0
    assert features["aa_C"] == 0.0


def test_invalid_sequence_raises_error():
    with pytest.raises(ValueError):
        amino_acid_composition("XYZ")