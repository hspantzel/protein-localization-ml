import pandas as pd


DATA_PATH = "data/raw/subcellular_location.tsv"


def main():
    df = pd.read_csv(DATA_PATH, sep="\t")

    # Remove the one row without a main location
    df = df.dropna(subset=["Main location"])

    print("Number of proteins with a main location:")
    print(len(df))

    print("\nReliability counts:")
    print(df["Reliability"].value_counts())

    # Some proteins can have more than one main location.
    # Split those locations into separate rows.
    locations = (
        df["Main location"]
        .str.split(";")
        .explode()
        .str.strip()
    )

    print("\nNumber of unique main locations:")
    print(locations.nunique())

    print("\nTop 20 most common locations:")
    print(locations.value_counts().head(20))


if __name__ == "__main__":
    main()