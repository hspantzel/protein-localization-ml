import pandas as pd


LABEL_PATH = "data/processed/protein_labels.csv"
SEQUENCE_PATH = "data/raw/uniprot_human_reviewed.tsv"
OUTPUT_PATH = "data/processed/protein_dataset.csv"


def main():
    # Load our cleaned HPA labels
    labels = pd.read_csv(LABEL_PATH)

    # Load UniProt protein sequences
    sequences = pd.read_csv(SEQUENCE_PATH, sep="\t")

    print(f"Genes in label dataset: {len(labels)}")
    print(f"UniProt protein entries: {len(sequences)}")

    # Remove entries without a gene name or sequence
    sequences = sequences.dropna(
        subset=["Gene Names (primary)", "Sequence"]
    ).copy()

    # Create standardized gene names for matching
    labels["gene_key"] = (
        labels["Gene name"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    sequences["gene_key"] = (
        sequences["Gene Names (primary)"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Calculate protein sequence length
    sequences["sequence_length"] = sequences["Sequence"].str.len()

    # If a gene appears more than once in UniProt,
    # keep the longest reviewed protein sequence
    sequences = (
        sequences
        .sort_values("sequence_length", ascending=False)
        .drop_duplicates(subset="gene_key", keep="first")
    )

    # Give columns cleaner names
    sequences = sequences.rename(
        columns={
            "Entry": "uniprot_accession",
            "Sequence": "sequence",
        }
    )

    # Match HPA genes with UniProt protein sequences
    merged = labels.merge(
        sequences[
            [
                "gene_key",
                "uniprot_accession",
                "sequence",
                "sequence_length",
            ]
        ],
        on="gene_key",
        how="inner",
    )

    print(f"\nMatched genes: {len(merged)}")

    match_rate = len(merged) / len(labels) * 100
    print(f"Match rate: {match_rate:.1f}%")

    print("\nSequence length statistics:")
    print(merged["sequence_length"].describe())

    # Show a few HPA genes that did not find a UniProt match
    matched_genes = set(merged["gene_key"])

    unmatched = labels[
        ~labels["gene_key"].isin(matched_genes)
    ]

    print(f"\nUnmatched genes: {len(unmatched)}")

    if len(unmatched) > 0:
        print("\nExample unmatched genes:")
        print(
            unmatched[["Gene", "Gene name"]]
            .head(10)
            .to_string(index=False)
        )

    # Remove temporary matching column
    merged = merged.drop(columns=["gene_key"])

    # Save final modeling dataset
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()