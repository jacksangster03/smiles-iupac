"""Training loop for SMILES <-> IUPAC translation."""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import ModelConfig
from model.transformer import Seq2SeqTransformer
from tokeniser.smiles_tokeniser import SmilesTokeniser
from tokeniser.iupac_tokeniser import IupacTokeniser


class TranslationDataset(Dataset):
    """Dataset of (source, target) encoded sequences."""

    def __init__(self, src_seqs, tgt_seqs):
        self.src_seqs = src_seqs
        self.tgt_seqs = tgt_seqs

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_seqs[idx], dtype=torch.long),
            torch.tensor(self.tgt_seqs[idx], dtype=torch.long),
        )


def load_pairs(tsv_path: str):
    """Load SMILES-IUPAC pairs from a TSV file."""
    pairs = []
    with open(tsv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def train(
    data_dir: str = "data",
    direction: str = "smiles2iupac",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_len: int = 200,
    max_pairs: int = 0,
    checkpoint_dir: str = "checkpoints",
):
    """Train the translation model.

    Args:
        direction: 'smiles2iupac' or 'iupac2smiles'
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_pairs = load_pairs(os.path.join(data_dir, "train.tsv"))
    val_pairs = load_pairs(os.path.join(data_dir, "val.tsv"))
    if max_pairs and len(train_pairs) > max_pairs:
        train_pairs = train_pairs[:max_pairs]
        val_pairs = val_pairs[: max(1000, max_pairs // 10)]
    print(f"Train: {len(train_pairs):,} pairs, Val: {len(val_pairs):,} pairs")

    # Build tokenisers
    smiles_tok = SmilesTokeniser()
    iupac_tok = IupacTokeniser()

    all_smiles = [p[0] for p in train_pairs]
    all_iupac = [p[1] for p in train_pairs]
    smiles_tok.build_vocab(all_smiles)
    iupac_tok.build_vocab(all_iupac)

    print(f"SMILES vocab: {smiles_tok.vocab_size}, IUPAC vocab: {iupac_tok.vocab_size}")

    # Save tokenisers
    smiles_tok.save(os.path.join(checkpoint_dir, "smiles_vocab.json"))
    iupac_tok.save(os.path.join(checkpoint_dir, "iupac_vocab.json"))

    # Determine source/target
    if direction == "smiles2iupac":
        src_tok, tgt_tok = smiles_tok, iupac_tok
        src_texts_train = [p[0] for p in train_pairs]
        tgt_texts_train = [p[1] for p in train_pairs]
        src_texts_val = [p[0] for p in val_pairs]
        tgt_texts_val = [p[1] for p in val_pairs]
    else:
        src_tok, tgt_tok = iupac_tok, smiles_tok
        src_texts_train = [p[1] for p in train_pairs]
        tgt_texts_train = [p[0] for p in train_pairs]
        src_texts_val = [p[1] for p in val_pairs]
        tgt_texts_val = [p[0] for p in val_pairs]

    # Encode
    print("Encoding sequences ...")
    src_train = [src_tok.encode(s, max_len=max_len) for s in src_texts_train]
    tgt_train = [tgt_tok.encode(s, max_len=max_len) for s in tgt_texts_train]
    src_val = [src_tok.encode(s, max_len=max_len) for s in src_texts_val]
    tgt_val = [tgt_tok.encode(s, max_len=max_len) for s in tgt_texts_val]

    train_ds = TranslationDataset(src_train, tgt_train)
    val_ds = TranslationDataset(src_val, tgt_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    config = ModelConfig(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        pad_idx=src_tok.pad_idx,
    )
    model = Seq2SeqTransformer(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_idx)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Warmup schedule
    warmup_steps = 4000
    step = 0

    def lr_lambda(s):
        s = max(s, 1)
        return min(s ** (-0.5), s * warmup_steps ** (-1.5)) * (config.d_model ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        t0 = time.time()

        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            src_pad_mask = src_batch == src_tok.pad_idx
            tgt_pad_mask = tgt_input == tgt_tok.pad_idx

            logits = model(src_batch, tgt_input, src_pad_mask, tgt_pad_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            step += 1

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                src_pad_mask = src_batch == src_tok.pad_idx
                tgt_pad_mask = tgt_input == tgt_tok.pad_idx
                logits = model(src_batch, tgt_input, src_pad_mask, tgt_pad_mask)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1),
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"Time: {elapsed:.0f}s | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": avg_val_loss,
                },
                os.path.join(checkpoint_dir, f"best_{direction}.pt"),
            )
            print(f"  Saved best model (val_loss={avg_val_loss:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--direction", default="smiles2iupac",
                        choices=["smiles2iupac", "iupac2smiles"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit training pairs (0 = use all)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        direction=args.direction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        max_pairs=args.max_pairs,
        checkpoint_dir=args.checkpoint_dir,
    )
