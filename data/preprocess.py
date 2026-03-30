"""Join PubChem CID-SMILES and CID-IUPAC, filter, and split.

Uses a two-pass streaming approach to avoid loading everything into memory:
1. First pass: read CID-IUPAC into a dict (smaller file, ~117M entries as strings)
2. Second pass: stream CID-SMILES, look up IUPAC, filter, and write directly
"""
import gzip
import os
import random

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
DATA_DIR = os.path.dirname(__file__)

MAX_SMILES_LEN = 150
MAX_IUPAC_LEN = 200
SAMPLE_SIZE = 500_000
SEED = 42


def preprocess():
    random.seed(SEED)

    # Pass 1: Build a set of CIDs from the SMALLER file first to know what to keep
    # Actually, let's use reservoir sampling to avoid storing all pairs
    # Stream both files, join on CID, and use reservoir sampling

    # Step 1: Load IUPAC CIDs into a dict (CID -> IUPAC name)
    # To save memory, only store CIDs where IUPAC name is short enough
    iupac_path = os.path.join(RAW_DIR, "CID-IUPAC.gz")
    print("Pass 1: Loading filtered IUPAC names ...")
    iupac_map = {}
    count = 0
    with gzip.open(iupac_path, "rt", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % 10_000_000 == 0:
                print(f"  Read {count:,} IUPAC lines, kept {len(iupac_map):,}")
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            if len(name) <= MAX_IUPAC_LEN:
                iupac_map[cid] = name

    print(f"  Kept {len(iupac_map):,} IUPAC names (len <= {MAX_IUPAC_LEN})")

    # Step 2: Stream SMILES, join with IUPAC, reservoir sample
    smiles_path = os.path.join(RAW_DIR, "CID-SMILES.gz")
    print("Pass 2: Streaming SMILES and joining ...")
    reservoir = []
    seen = 0
    with gzip.open(smiles_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            smi = parts[1]
            if len(smi) > MAX_SMILES_LEN:
                continue
            if cid not in iupac_map:
                continue

            pair = (smi, iupac_map[cid])
            seen += 1

            # Reservoir sampling
            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(pair)
            else:
                j = random.randint(0, seen - 1)
                if j < SAMPLE_SIZE:
                    reservoir[j] = pair

            if seen % 5_000_000 == 0:
                print(f"  Matched {seen:,} pairs so far")

    print(f"Total matched pairs: {seen:,}")
    print(f"Reservoir size: {len(reservoir):,}")

    # Free memory
    del iupac_map

    # Shuffle and split 90/5/5
    random.shuffle(reservoir)
    n = len(reservoir)
    train_end = int(0.9 * n)
    val_end = int(0.95 * n)

    splits = {
        "train": reservoir[:train_end],
        "val": reservoir[train_end:val_end],
        "test": reservoir[val_end:],
    }

    for split_name, split_pairs in splits.items():
        path = os.path.join(DATA_DIR, f"{split_name}.tsv")
        with open(path, "w") as f:
            f.write("smiles\tiupac\n")
            for smi, name in split_pairs:
                f.write(f"{smi}\t{name}\n")
        print(f"Saved {len(split_pairs):,} pairs to {path}")


if __name__ == "__main__":
    preprocess()
