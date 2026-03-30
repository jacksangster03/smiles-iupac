"""Regex-based SMILES tokeniser using the Schwaller pattern.

Splits SMILES strings into chemically meaningful tokens:
    'BrC(=O)c1ccccc1' -> ['Br', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1']
"""
import re
from typing import List, Dict, Optional

# Schwaller regex: matches two-letter elements first, then single chars
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%\d{2}|\d)"
)

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class SmilesTokeniser:
    """Tokeniser for SMILES strings."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.token_to_idx = vocab
        else:
            self.token_to_idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx["<pad>"]

    @property
    def sos_idx(self) -> int:
        return self.token_to_idx["<sos>"]

    @property
    def eos_idx(self) -> int:
        return self.token_to_idx["<eos>"]

    @property
    def unk_idx(self) -> int:
        return self.token_to_idx["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def tokenise(self, smiles: str) -> List[str]:
        """Split a SMILES string into tokens."""
        tokens = SMILES_REGEX.findall(smiles)
        return tokens

    def encode(self, smiles: str, max_len: Optional[int] = None) -> List[int]:
        """Tokenise and convert to integer indices. Adds <sos> and <eos>."""
        tokens = self.tokenise(smiles)
        indices = [self.sos_idx]
        for tok in tokens:
            indices.append(self.token_to_idx.get(tok, self.unk_idx))
        indices.append(self.eos_idx)

        if max_len is not None:
            indices = indices[:max_len]
            indices += [self.pad_idx] * (max_len - len(indices))

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert integer indices back to a SMILES string."""
        tokens = []
        for idx in indices:
            tok = self.idx_to_token.get(idx, "<unk>")
            if tok == "<eos>":
                break
            if tok in ("<pad>", "<sos>"):
                continue
            tokens.append(tok)
        return "".join(tokens)

    def build_vocab(self, smiles_list: List[str]):
        """Build vocabulary from a list of SMILES strings."""
        self.token_to_idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for smiles in smiles_list:
            for tok in self.tokenise(smiles):
                if tok not in self.token_to_idx:
                    self.token_to_idx[tok] = len(self.token_to_idx)
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

    def save(self, path: str):
        """Save vocabulary to a file."""
        import json
        with open(path, "w") as f:
            json.dump(self.token_to_idx, f)

    @classmethod
    def load(cls, path: str) -> "SmilesTokeniser":
        """Load vocabulary from a file."""
        import json
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
