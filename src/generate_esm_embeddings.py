import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DATA_PATH = Path("data/processed/protein_dataset.csv")

EMBEDDING_PATH = Path(
    "data/processed/esm2_embeddings.npy"
)

METADATA_PATH = Path(
    "data/processed/esm2_metadata.csv"
)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# ESM-2 has a sequence length limit.
# We keep the first 1022 amino acids for long proteins.
MAX_RESIDUES = 1022


def clean_sequence(sequence):
    sequence = str(sequence).strip().upper()

    if len(sequence) <= MAX_RESIDUES:
        return sequence

    # For very long proteins, preserve both ends of the sequence.
    # Localization signals can occur near the N- or C-terminus.
    half = MAX_RESIDUES // 2

    return sequence[:half] + sequence[-half:]


def mean_pool_embeddings(hidden_states, attention_mask):
    """
    Average residue embeddings while excluding
    special beginning/end tokens and padding.
    """

    token_mask = attention_mask.clone()

    # Ignore beginning-of-sequence token
    token_mask[:, 0] = 0

    # Ignore end-of-sequence token
    lengths = attention_mask.sum(dim=1)

    for index, length in enumerate(lengths):
        token_mask[index, length - 1] = 0

    token_mask = token_mask.unsqueeze(-1)

    summed = (hidden_states * token_mask).sum(dim=1)

    counts = token_mask.sum(dim=1).clamp(min=1)

    return summed / counts


def main(limit=None, batch_size=4):
    print("Loading protein dataset...")

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit).copy()

    print(f"Proteins to embed: {len(df)}")

    print("\nLoading ESM-2 model:")
    print(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    device = torch.device("cpu")
    model.to(device)

    embeddings = []

    print("\nGenerating embeddings...")

    for start in tqdm(
        range(0, len(df), batch_size)
    ):
        batch = df.iloc[
            start:start + batch_size
        ]

        sequences = [
            clean_sequence(sequence)
            for sequence in batch["sequence"]
        ]

        inputs = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pool_embeddings(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        )

        embeddings.append(
            pooled.cpu().numpy()
        )

    embedding_matrix = np.vstack(embeddings)

    np.save(
        EMBEDDING_PATH,
        embedding_matrix,
    )

    metadata_columns = [
        "Gene",
        "Gene name",
        "uniprot_accession",
        "sequence_length",
        "nucleoplasm",
        "cytosol",
        "vesicles",
        "plasma_membrane",
        "mitochondria",
        "golgi_apparatus",
        "nucleoli",
        "endoplasmic_reticulum",
    ]

    df[metadata_columns].to_csv(
        METADATA_PATH,
        index=False,
    )

    print("\nFinished.")
    print(
        f"Embedding shape: "
        f"{embedding_matrix.shape}"
    )

    print(
        f"Saved embeddings to: "
        f"{EMBEDDING_PATH}"
    )

    print(
        f"Saved metadata to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    main(
        limit=args.limit,
        batch_size=args.batch_size,
    )