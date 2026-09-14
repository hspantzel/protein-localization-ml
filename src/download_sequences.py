from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


OUTPUT_PATH = Path("data/raw/uniprot_human_reviewed.tsv")

BASE_URL = "https://rest.uniprot.org/uniprotkb/stream"

PARAMS = {
    "format": "tsv",
    "fields": "accession,gene_primary,sequence",
    "query": "(reviewed:true) AND (organism_id:9606)",
}


def main():
    url = f"{BASE_URL}?{urlencode(PARAMS)}"

    request = Request(
        url,
        headers={"User-Agent": "protein-localization-ml/1.0"},
    )

    print("Downloading reviewed human protein sequences from UniProt...")

    with urlopen(request, timeout=120) as response:
        with open(OUTPUT_PATH, "wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                output_file.write(chunk)

    print(f"Saved to: {OUTPUT_PATH}")

    df = pd.read_csv(OUTPUT_PATH, sep="\t")

    print(f"\nNumber of protein entries: {len(df)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()