"""Character-level IUPAC tokeniser with option to train BPE.

Starts character-level for the MVP, with a BPE upgrade path.
"""
from typing import List, Dict, Optional

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# Characters commonly found in IUPAC names
IUPAC_CHARS = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-,;:.()[]{}'/\\+ "
)


class IupacTokeniser:
    """Character-level tokeniser for IUPAC names."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.token_to_idx = vocab
        else:
            self.token_to_idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
            for ch in IUPAC_CHARS:
                if ch not in self.token_to_idx:
                    self.token_to_idx[ch] = len(self.token_to_idx)
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

    def tokenise(self, name: str) -> List[str]:
        """Split an IUPAC name into characters."""
        return list(name)

    def encode(self, name: str, max_len: Optional[int] = None) -> List[int]:
        """Encode an IUPAC name to integer indices. Adds <sos> and <eos>."""
        tokens = self.tokenise(name)
        indices = [self.sos_idx]
        for ch in tokens:
            indices.append(self.token_to_idx.get(ch, self.unk_idx))
        indices.append(self.eos_idx)

        if max_len is not None:
            indices = indices[:max_len]
            indices += [self.pad_idx] * (max_len - len(indices))

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert integer indices back to an IUPAC name."""
        chars = []
        for idx in indices:
            tok = self.idx_to_token.get(idx, "?")
            if tok == "<eos>":
                break
            if tok in ("<pad>", "<sos>"):
                continue
            chars.append(tok)
        return "".join(chars)

    def build_vocab(self, names: List[str]):
        """Extend vocabulary from a list of IUPAC names."""
        for name in names:
            for ch in name:
                if ch not in self.token_to_idx:
                    self.token_to_idx[ch] = len(self.token_to_idx)
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

    def save(self, path: str):
        """Save vocabulary to a file."""
        import json
        with open(path, "w") as f:
            json.dump(self.token_to_idx, f)

    @classmethod
    def load(cls, path: str) -> "IupacTokeniser":
        """Load vocabulary from a file."""
        import json
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
