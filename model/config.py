"""Hyperparameter configuration."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    src_vocab_size: int = 100
    tgt_vocab_size: int = 100
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 200
    pad_idx: int = 0
