import pandas as pd


INPUT_PATH = "data/raw/subcellular_location.tsv"
OUTPUT_PATH = "data/processed/protein_labels.csv"

TARGET_LOCATIONS = [
    "Nucleoplasm",
    "Cytosol",
    "Vesicles",
    "Plasma membrane",
    "Mitochondria",
    "Golgi apparatus",
    "Nucleoli",
    "Endoplasmic reticulum",
]


def split_locations(value):
    """Split a semicolon-separated location string."""
    return [location.strip() for location in value.split(";")]


def main():
    df = pd.read_csv(INPUT_PATH, sep="\t")

    # Remove rows without a main location
    df = df.dropna(subset=["Main location"]).copy()

    # Remove uncertain annotations
    df = df[df["Reliability"] != "Uncertain"].copy()

    # Turn the Main location column into Python lists
    df["locations"] = df["Main location"].apply(split_locations)

    # Keep only the locations we want to predict
    df["target_locations"] = df["locations"].apply(
        lambda locations: [
            location
            for location in locations
            if location in TARGET_LOCATIONS
        ]
    )

    # Remove genes that have none of our target locations
    df = df[df["target_locations"].str.len() > 0].copy()

    # Create one binary column for every target location
    for location in TARGET_LOCATIONS:
        column_name = (
            location.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        df[column_name] = df["target_locations"].apply(
            lambda locations: int(location in locations)
        )

    # Convert list to a readable text representation
    df["target_locations"] = df["target_locations"].apply(
        lambda locations: "|".join(locations)
    )

    output_columns = [
        "Gene",
        "Gene name",
        "Reliability",
        "target_locations",
    ]

    output_columns += [
        location.lower().replace(" ", "_").replace("-", "_")
        for location in TARGET_LOCATIONS
    ]

    output_df = df[output_columns]

    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print(f"Number of genes: {len(output_df)}")

    print("\nClass counts:")

    for location in TARGET_LOCATIONS:
        column_name = location.lower().replace(" ", "_").replace("-", "_")
        print(f"{location}: {output_df[column_name].sum()}")


if __name__ == "__main__":
    main()