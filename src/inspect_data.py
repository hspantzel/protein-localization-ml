import pandas as pd


DATA_PATH = "data/raw/subcellular_location.tsv"


def main():
    df = pd.read_csv(DATA_PATH, sep="\t")

    print("Dataset shape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()