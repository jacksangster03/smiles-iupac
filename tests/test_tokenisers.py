"""Tests for SMILES and IUPAC tokenisers."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokeniser.smiles_tokeniser import SmilesTokeniser
from tokeniser.iupac_tokeniser import IupacTokeniser


def test_smiles_tokenise():
    tok = SmilesTokeniser()
    tokens = tok.tokenise("BrC(=O)c1ccccc1")
    assert tokens == ["Br", "C", "(", "=", "O", ")", "c", "1", "c", "c", "c", "c", "c", "1"]


def test_smiles_two_letter_elements():
    tok = SmilesTokeniser()
    tokens = tok.tokenise("ClCCBr")
    assert "Cl" in tokens
    assert "Br" in tokens


def test_smiles_roundtrip():
    tok = SmilesTokeniser()
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    tok.build_vocab([smiles])
    encoded = tok.encode(smiles)
    decoded = tok.decode(encoded)
    assert decoded == smiles


def test_smiles_padding():
    tok = SmilesTokeniser()
    tok.build_vocab(["CC"])
    encoded = tok.encode("CC", max_len=10)
    assert len(encoded) == 10
    assert encoded[-1] == tok.pad_idx


def test_iupac_roundtrip():
    tok = IupacTokeniser()
    name = "2-methylpropan-1-ol"
    encoded = tok.encode(name)
    decoded = tok.decode(encoded)
    assert decoded == name


def test_iupac_padding():
    tok = IupacTokeniser()
    encoded = tok.encode("ethanol", max_len=20)
    assert len(encoded) == 20


def test_iupac_unknown_char():
    tok = IupacTokeniser()
    # Greek letter not in default vocab
    encoded = tok.encode("alpha-\u03b1")
    decoded = tok.decode(encoded)
    # Unknown chars get <unk> idx; decoded string won't match original
    assert decoded != "alpha-\u03b1"
    assert tok.unk_idx in encoded


if __name__ == "__main__":
    test_smiles_tokenise()
    test_smiles_two_letter_elements()
    test_smiles_roundtrip()
    test_smiles_padding()
    test_iupac_roundtrip()
    test_iupac_padding()
    test_iupac_unknown_char()
    print("All tests passed!")
