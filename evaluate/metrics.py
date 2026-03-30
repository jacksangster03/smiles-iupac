"""Evaluation metrics for SMILES <-> IUPAC translation."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import ModelConfig
from model.transformer import Seq2SeqTransformer
from model.train import load_pairs
from tokeniser.smiles_tokeniser import SmilesTokeniser
from tokeniser.iupac_tokeniser import IupacTokeniser


def exact_match(predictions: list, references: list) -> float:
    """Fraction of predictions that exactly match the reference."""
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions) if predictions else 0.0


def smiles_validity(smiles_list: list) -> float:
    """Fraction of SMILES strings that RDKit can parse."""
    try:
        from rdkit import Chem
    except ImportError:
        print("RDKit not available, skipping validity check")
        return -1.0

    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0.0


def canonical_match(pred_smiles: list, ref_smiles: list) -> float:
    """Fraction where canonicalised predicted SMILES matches canonicalised reference."""
    try:
        from rdkit import Chem
    except ImportError:
        return -1.0

    matches = 0
    for pred, ref in zip(pred_smiles, ref_smiles):
        pred_mol = Chem.MolFromSmiles(pred)
        ref_mol = Chem.MolFromSmiles(ref)
        if pred_mol is not None and ref_mol is not None:
            if Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(ref_mol):
                matches += 1
    return matches / len(pred_smiles) if pred_smiles else 0.0


def evaluate(
    checkpoint_path: str,
    data_dir: str = "data",
    direction: str = "smiles2iupac",
    max_len: int = 200,
    num_samples: int = 1000,
):
    """Run evaluation on the test set."""
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Load tokenisers
    ckpt_dir = os.path.dirname(checkpoint_path)
    smiles_tok = SmilesTokeniser.load(os.path.join(ckpt_dir, "smiles_vocab.json"))
    iupac_tok = IupacTokeniser.load(os.path.join(ckpt_dir, "iupac_vocab.json"))

    if direction == "smiles2iupac":
        src_tok, tgt_tok = smiles_tok, iupac_tok
    else:
        src_tok, tgt_tok = iupac_tok, smiles_tok

    # Load model
    model = Seq2SeqTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_pairs = load_pairs(os.path.join(data_dir, "test.tsv"))
    if num_samples < len(test_pairs):
        test_pairs = test_pairs[:num_samples]

    if direction == "smiles2iupac":
        src_texts = [p[0] for p in test_pairs]
        ref_texts = [p[1] for p in test_pairs]
    else:
        src_texts = [p[1] for p in test_pairs]
        ref_texts = [p[0] for p in test_pairs]

    # Translate
    predictions = []
    batch_size = 32
    for i in range(0, len(src_texts), batch_size):
        batch_texts = src_texts[i : i + batch_size]
        src_encoded = [src_tok.encode(s, max_len=max_len) for s in batch_texts]
        src_tensor = torch.tensor(src_encoded, dtype=torch.long, device=device)
        src_pad_mask = src_tensor == src_tok.pad_idx

        output = model.translate(
            src_tensor, src_pad_mask, tgt_tok.sos_idx, tgt_tok.eos_idx, max_len
        )

        for seq in output:
            pred = tgt_tok.decode(seq.tolist())
            predictions.append(pred)

    # Compute metrics
    em = exact_match(predictions, ref_texts)
    print(f"Exact match: {em:.4f} ({int(em * len(predictions))}/{len(predictions)})")

    if direction == "iupac2smiles":
        validity = smiles_validity(predictions)
        print(f"SMILES validity: {validity:.4f}")
        cm = canonical_match(predictions, ref_texts)
        print(f"Canonical match: {cm:.4f}")

    # Show examples
    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        print(f"  SRC: {src_texts[i]}")
        print(f"  REF: {ref_texts[i]}")
        print(f"  PRD: {predictions[i]}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--direction", default="smiles2iupac",
                        choices=["smiles2iupac", "iupac2smiles"])
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.direction, num_samples=args.num_samples)
