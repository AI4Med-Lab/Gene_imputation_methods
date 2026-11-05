import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from typing import Optional
import logging
import anndata as ad

# =========================================================
# 2️⃣ Encoder Block
# =========================================================
class EncoderBlock(nn.Module):
    """Encodes ST or scRNA features into latent representations."""
    def __init__(self, in_dim, emb_dim=256, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 3️⃣ Cross-Attention Decoder
# =========================================================
class CrossAttentionDecoder(nn.Module):
    """Multi-head cross-attention block for decoding ST queries using sc centroids."""
    def __init__(self, emb_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, per_head: bool = False):
        """
+        query: (B, L_q, emb_dim)
+        key/value: (B, L_k, emb_dim)
+        per_head: if True, return per-head weights shape (B, num_heads, L_q, L_k),
+                  else return averaged weights shape (B, L_q, L_k).
+        """
        attn_out, attn_weights = self.attn(query, key, value, need_weights=True, average_attn_weights=not per_head)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_weights


# =========================================================
# 4️⃣ Attention Imputer Model
# =========================================================
class Casper(nn.Module):
    """
    Full model: Encoder (ST) -> Decoder (scRNA centroids) with cross-attention.
    """
    def __init__(self, st_dim, sc_dim, out_dim, emb_dim=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.encoder_st = EncoderBlock(in_dim=st_dim, emb_dim=emb_dim, dropout=dropout)
        self.encoder_sc = EncoderBlock(in_dim=sc_dim, emb_dim=emb_dim, dropout=dropout)
        self.decoder = CrossAttentionDecoder(emb_dim=emb_dim, n_heads=n_heads, dropout=dropout)
        self.fc_out = nn.Linear(emb_dim, out_dim)

    def forward(self, st_input: torch.Tensor, sc_centroids: torch.Tensor, positive_x=None, negative_x=None, per_head: int  = False):

        B = st_input.size(0)

        if not torch.is_tensor(sc_centroids):
            sc_centroids = torch.tensor(sc_centroids, dtype=st_input.dtype, device=st_input.device)
        else:
            sc_centroids = sc_centroids.to(st_input.device, dtype=st_input.dtype)

        # Encode ST (queries)
        Q = self.encoder_st(st_input).unsqueeze(1)  # (B, 1, emb_dim)


        if positive_x is not None and negative_x is not None:
            emb_anchor  = Q.squeeze()
            emb_positive = self.encoder_st(positive_x)
            emb_negative = self.encoder_st(negative_x)


        # Encode scRNA centroids (keys, values)
        KV = self.encoder_sc(sc_centroids)          # (n_centroids, emb_dim)
        KV = KV.unsqueeze(0).expand(B, -1, -1)      # (B, n_centroids, emb_dim)

        # Cross-attention decoding (optionally return per-head weights)
        attn_out, attn_weights = self.decoder(Q, KV, KV, per_head=per_head)

        print(f"attn_out shape =========== {attn_out.shape}")
        print(f"attn_out shape =========== {attn_out.squeeze(1).shape}")

        out = self.fc_out(attn_out.squeeze(1))      # (B, out_dim)
        if positive_x is not None and negative_x is not None:
            return out, attn_weights, emb_anchor, emb_positive, emb_negative
        else:
            return out, attn_weights


