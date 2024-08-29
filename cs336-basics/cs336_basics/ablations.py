#!/usr/bin/env python3

from typing import Optional, Type

import torch
from torch import nn
import torch.nn.functional as F

from cs336_basics.model import(
    RMSNorm,
    PositionwiseFeedforward,
    MultiheadSelfAttention
)


# CLASSES

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
        eps: float = 1e-5,
        device: str = "cpu"
    ):

        super().__init__()
        self.num_layers = num_layers
        self.device = device
        self.token_embeddings = nn.Embedding(vocab_size, d_model).to(device)
        self.position_embeddings = nn.Embedding(context_length, d_model).to(device) 
        self.dropout = nn.Dropout(residual_pdrop).to(device) if residual_pdrop is not None else nn.Identity().to(device) 

        # transformer blocks
        self.layers = nn.ModuleList([
            PreNormAblation(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                weights=weights,
                attn_pdrop=attn_pdrop,
                residual_pdrop=residual_pdrop,
                eps=eps
            ) for _ in range(self.num_layers)
        ])

        self.ln_final = RMSNorm(d_model, eps=eps).to(device) 
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device) 

    def forward(self, in_indices: torch.LongTensor) -> torch.FloatTensor:
        # Get token embeddings
        token_emb = self.token_embeddings(in_indices.to(self.device)).to(self.device)  # Shape: (batch_size, sequence_length, d_model)
        
        # Generate position indices
        positions = torch.arange(in_indices.size(1)).unsqueeze(0).to(self.device) 
        
        # Get position embeddings
        position_emb = self.position_embeddings(positions).to(self.device)  # Shape: (1, sequence_length, d_model)
        
        # Add position embeddings to token embeddings
        combined_emb = (token_emb + position_emb).to(self.device)   # Shape: (batch_size, sequence_length, d_model)

        x = self.dropout(combined_emb).to(self.device) 

        # transformer blocks
        for layer in self.layers:
            x = layer(x)

        # normalize
        x = self.ln_final(x)

        # output embeddings
        x = self.lm_head(x)

        return x
    

class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
        eps: float = 1e-5
        ):

        super().__init__()
        self.ln = RMSNorm(d_model, eps=eps)

        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        self.dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        residual = in_features
        x = in_features
        x = self.ln(x)
        x = self.attn(x) + self.ffn(x)
        x = self.dropout(x)
        x += residual

        return x
    

class LayerNormAblation(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
        eps: float = 1e-5
        ):

        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        self.dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        # multi-head self-attention
        residual = in_features
        x = in_features
        x = self.attn(x)
        x = self.dropout(x)
        x += residual

        # feed forward
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual

        return x
    

class PreNormAblation(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        weights: dict[str, torch.FloatTensor],
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
        eps: float = 1e-5
        ):

        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps)
        self.ln2 = RMSNorm(d_model, eps=eps)

        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        self.dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        # multi-head self-attention
        residual = in_features
        x = in_features
        x = self.attn(x)
        x = self.dropout(x)
        x += residual
        x = self.ln1(x)

        # feed forward
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual
        x = self.ln2(x)

        return x