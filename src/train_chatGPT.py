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
