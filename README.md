# SMILES-IUPAC Bi-directional Translation

A lightweight encoder-decoder transformer for bi-directional translation between SMILES strings and IUPAC chemical names. Built as a prototype for the [DeepChem GSoC 2026](https://github.com/deepchem/deepchem/discussions/4703) project on robust chemical name-structure conversion.

## Architecture

- **Model**: Encoder-decoder transformer (PyTorch `nn.Transformer`)
  - 4 encoder layers, 4 decoder layers, 8 attention heads, d_model=256
  - Sinusoidal positional encoding, teacher forcing during training
  - Greedy decoding for inference (beam search planned)
  - Warmup learning rate schedule (4000 warmup steps)

- **SMILES Tokeniser**: Regex-based using the [Schwaller pattern](https://doi.org/10.1039/C8SC02339E) — splits SMILES into chemically meaningful tokens (two-letter elements, bracket atoms, ring closures, stereochemistry markers). Vocab: ~242 tokens.

- **IUPAC Tokeniser**: Character-level for MVP (vocab: ~84 tokens). Word-level tokeniser with IUPAC-aware regex planned for the full implementation.

## Data

- **Source**: PubChem CID-SMILES and CID-IUPAC (public domain)
- **Size**: 500,000 pairs (reservoir sampled from ~90M+ matched entries)
- **Filters**: SMILES length ≤ 150, IUPAC length ≤ 200
- **Split**: 90/5/5 train/val/test (seed=42)

## Usage

### Setup

```bash
conda env create -f environment.yml
conda activate smiles-iupac
```

### Download and preprocess data

```bash
python data/download.py      # downloads PubChem files (~2.5 GB)
python data/preprocess.py    # joins, filters, samples, splits
```

### Train

```bash
# SMILES -> IUPAC
python model/train.py --direction smiles2iupac --epochs 10 --batch-size 64

# IUPAC -> SMILES
python model/train.py --direction iupac2smiles --epochs 10 --batch-size 64

# Quick test with limited data
python model/train.py --max-pairs 10000 --epochs 5
```

### Evaluate

```bash
python evaluate/metrics.py --checkpoint checkpoints/best_smiles2iupac.pt --direction smiles2iupac
```

### Run tests

```bash
python -m pytest tests/
```

## Project Structure

```
smiles-iupac/
├── data/
│   ├── download.py          # PubChem FTP download script
│   └── preprocess.py        # Two-pass streaming join + reservoir sampling
├── model/
│   ├── config.py            # ModelConfig dataclass
│   ├── train.py             # Training loop with warmup schedule
│   └── transformer.py       # Seq2SeqTransformer (encoder-decoder)
├── tokeniser/
│   ├── smiles_tokeniser.py  # Schwaller regex tokeniser
│   └── iupac_tokeniser.py   # Character-level tokeniser
├── evaluate/
│   └── metrics.py           # Exact match, SMILES validity, canonical match
├── tests/
│   └── test_tokenisers.py   # Unit tests for both tokenisers
├── checkpoints/             # Saved vocabularies and model weights
├── notebooks/               # (planned) analysis and demo notebooks
├── scripts/                 # (planned) benchmark and utility scripts
└── environment.yml          # Conda environment specification
```

## Planned Improvements (GSoC Coding Period)

- Word-level IUPAC tokeniser with chemistry-aware regex
- Beam search decoding with length normalisation
- SMILES augmentation via RDKit random enumeration
- DeepChem integration (`TorchModel` subclass)
- Comprehensive benchmarking against OPSIN, STOUT, and frontier LLMs
- Pre-trained weights and Colab tutorial

## Licence

MIT
