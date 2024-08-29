#!/usr/bin/env python3

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from cs336_systems.kernel import TritonRMSNormFunction

## CLASSES

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
        attn_pdrop: Optional[float] = 0.1,
        residual_pdrop: Optional[float] = 0.1,
        eps: float = 1e-5,
        device: str = "cpu"
    ):

        super().__init__()
        self.rms_norm_weights = nn.Parameter(torch.randn(d_model))

        self.num_layers = num_layers
        self.device = device
        self.token_embeddings = nn.Embedding(vocab_size, d_model).to(device)
        self.position_embeddings = nn.Embedding(context_length, d_model).to(device) 
        self.dropout = nn.Dropout(residual_pdrop).to(device) if residual_pdrop is not None else nn.Identity().to(device) 

        # transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                weights=weights,
                attn_pdrop=attn_pdrop,
                residual_pdrop=residual_pdrop,
                eps=eps
            ) for _ in range(self.num_layers)
        ])

        # self.ln_final = nn.LayerNorm(d_model, eps=eps).to(device) 
        # self.ln_final = RMSNorm(d_model, eps=eps).to(device) 
        self.ln_final = TritonRMSNormFunction.apply
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
        # x = self.ln_final(x)
        x = self.ln_final(x, self.rms_norm_weights)

        # output embeddings
        x = self.lm_head(x)

        return x


class TransformerBlock(nn.Module):
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
        self.rms_norm_weights = nn.Parameter(torch.randn(d_model))

        # self.ln1 = RMSNorm(d_model, eps=eps)
        # self.ln2 = RMSNorm(d_model, eps=eps)
        # self.ln1 = nn.LayerNorm(d_model, elementwise_affine=True)
        # self.ln2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ln1 = TritonRMSNormFunction.apply
        self.ln2 = TritonRMSNormFunction.apply

        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedforward(d_model, d_ff)
        self.dropout = nn.Dropout(residual_pdrop) if residual_pdrop is not None else nn.Identity()

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        # multi-head self-attention
        residual = in_features
        x = in_features
        # x = self.ln1(x)
        x = self.ln1(x, self.rms_norm_weights)
        x = self.attn(x)
        x = self.dropout(x)
        x += residual

        # feed forward
        residual = x
        # x = self.ln2(x)
        x = self.ln2(x, self.rms_norm_weights)
        x = self.ffn(x)
        x = self.dropout(x)
        x += residual

        return x


class RMSNorm(nn.Module):
    """
    Root mean square layer normalization for layer normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        mean_square = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + self.eps)
        out_features = in_features / mean_square * self.weight
        return out_features


class PositionwiseFeedforward(nn.Module):
    """
    Position-wise feed-forward network linear transformation, omitting biases.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        return self.w2(gelu(self.w1(in_features)))


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        q_reshape = self._tensor_reshape(self.q_proj(in_features), True)
        k_reshape = self._tensor_reshape(self.k_proj(in_features), True)
        v_reshape = self._tensor_reshape(self.v_proj(in_features), True)

        # calculate causal attention mask
        seq_len = in_features.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # compute attention output with causal mask
        attention_reshaped = scaled_dot_product_attention(q_reshape, k_reshape, v_reshape, mask=causal_mask, pdrop=self.attn_pdrop)

        # rearrange into concatenated
        attention = self._tensor_reshape(attention_reshaped, False)

        # apply output projection
        return self.output_proj(attention)

    def _tensor_reshape(self, x: torch.FloatTensor, flag: bool) -> torch.FloatTensor:
        if flag:  
            # reshape and transpose to (batch_dims, num_heads, seq_len, -1)
            batch_dims = x.dim() - 1  # number of dimensions before d_model
            batch_shape = x.shape[:batch_dims]
            x = x.view(*batch_shape, self.num_heads, x.shape[-1] // self.num_heads)
            x = x.transpose(-2, -3)
        else:
            # revert back to (batch_dims, seq_len, d_model)
            x = x.transpose(-2, -3)
            batch_dims = x.dim() - 2  # number of dimensions before head, d_model
            batch_shape = x.shape[:batch_dims]
            x = x.reshape(*batch_shape, x.shape[-1] * self.num_heads)

        return x
    
        
## FUNCTIONS

def gelu(x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Gaussian Error Linear Unit activiation function.
    """
    return x / 2 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


def scaled_dot_product_attention(
    Q: torch.FloatTensor,
    K: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
    eps: float = 1e-5
) -> torch.FloatTensor:
    """
    Given key (K), query (Q), and value (V) tensors, 
    return output of  scaled dot product attention.
    """
    d_k = K.shape[-1]
    score = (Q @ K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k) + eps)

    if mask is not None:
        mask = mask.to(Q.device)
        score += -1e15 * mask
    attention_weight = F.softmax(score, dim=-1)
    
    if pdrop is not None:
        dropout_layer = nn.Dropout(p=pdrop)
        attention_weight_dropout = dropout_layer(attention_weight)
    attention = attention_weight_dropout @ V
    
    return attention
