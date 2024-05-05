"""Modules for model resembling chatGPT. 
Implementation closely mirrors https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper-parameters
block_size = 64
batch_size = 16
num_heads = 4
n_layers = 4
emb_dim = 64
num_iters = 5000
eval_iters = 200
max_output_size = 1500
dropout = 0.0

# Use GPU if available
devide = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """Single attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(emb_dim, self.head_size, bias=False)
        self.key = nn.Linear(emb_dim, self.head_size, bias=False)
        self.value = nn.Linear(emb_dim, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # shape of x: bxtxemb_dim
        q = self.query(x)  # bxtxhead_size
        k = self.key(x)  # same as above
        v = self.value(x)  # same as above

        # Computing self-attention scores
        weights = (q @ k.transpose(1, 2)) / (
            k.shape[-1] ** 0.5
        )  # bxtxhead_size @ bxhead_sizext -> bxtxt
        # Decoder only model so masking future tokens
        weights.masked_fill_(
            self.tril[:T, :T] == 0, float("-inf")
        )  # slicing to length token is necessary for variable sequence length
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        # Scale the value vectors with self-attention weights
        context = weights @ v  # bxtxhead_size
        return context


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.heads = nn.ModuleList(Head(self.head_size) for _ in range(num_heads))
        self.proj = nn.Linear(self.head_size * 4, self.head_size * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate context vector for all heads
        grouped_attention_weights = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # bxtxhead_size
        # Linear projection
        out = self.dropout(self.proj(grouped_attention_weights))
        return out  # bxtxhead_size*4
