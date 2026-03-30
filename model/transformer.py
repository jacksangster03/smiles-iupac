"""Encoder-decoder transformer for sequence-to-sequence translation."""
import math
import torch
import torch.nn as nn
from .config import ModelConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """Transformer encoder-decoder for SMILES <-> IUPAC translation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(
            config.src_vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(config.d_model, config.tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token indices
            tgt: (batch, tgt_len) target token indices
            src_key_padding_mask: (batch, src_len) True where padded
            tgt_key_padding_mask: (batch, tgt_len) True where padded

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device
        )

        src_emb = self.pos_encoding(
            self.src_embedding(src) * math.sqrt(self.config.d_model)
        )
        tgt_emb = self.pos_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.config.d_model)
        )

        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 200,
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_emb = self.pos_encoding(
            self.src_embedding(src) * math.sqrt(self.config.d_model)
        )
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # Start with <sos>
        tgt_indices = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoding(
                self.tgt_embedding(tgt_indices) * math.sqrt(self.config.d_model)
            )
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_indices.size(1), device=device
            )

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            next_token_logits = self.output_proj(output[:, -1, :])
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            tgt_indices = torch.cat([tgt_indices, next_token], dim=1)

            # Stop if all sequences have generated <eos>
            if (next_token == eos_idx).all():
                break

        return tgt_indices
