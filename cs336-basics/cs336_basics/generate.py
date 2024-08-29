#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from cs336_basics.tokenizer import Tokenizer


# FUNCTIONS

def softmax(tensor: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    # subtract the largest element for numerical stability
    max_vals = torch.max(tensor, dim=dim, keepdim=True).values
    tensor -= max_vals

    # compute softmax
    exp_inputs = torch.exp(tensor)
    return exp_inputs / torch.sum(exp_inputs, dim=dim, keepdim=True)


def top_p_sampling(logits: torch.FloatTensor, top_p: int) -> torch.FloatTensor:
    """
    Apply top-p sampling to the logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum((sorted_logits), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False
    sorted_logits[sorted_indices_to_remove] = 0
    return sorted_logits / torch.sum(sorted_logits, dim=-1, keepdim=True)


def decode(prompt, train_model, max_tokens=50, temperature=1.0, top_p=0.9):
    """
    Generate text completion for the given prompt.
    """
    generated_tokens = []
    input_ids = torch.tensor(Tokenizer.encode(prompt)).unsqueeze(0).to(train_model.device)
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = train_model.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            sampled_index = top_p_sampling(softmax(logits / temperature), top_p)
            generated_tokens.append(sampled_index.item())
            input_ids = torch.cat([input_ids, sampled_index.unsqueeze(0)], dim=-1)
            if sampled_index == train_model.tokenizer.eos_token_id:  # End of sequence token
                break
    generated_text = train_model.tokenizer.decode(generated_tokens)
    return generated_text
