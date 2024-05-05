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
lr = 0.001
dropout = 0.0

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download data in not already
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read()

print(f"Total number of characters in the dataset: {len(text)}")

# All unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character level tokenizer
stoi = {char: idx for idx, char in enumerate(chars)}
itos = {idx: char for char, idx in stoi.items()}

encode = lambda s: [stoi[char] for char in s]
decode = lambda i: "".join([itos[idx] for idx in i])

# Tokenize all text
data = torch.tensor(encode(text), dtype=torch.long)


def train_test_split(data, test_pct=0.1):
    """Split data to train and test at a given percentage"""
    data_size = len(data)
    num_test_samples = int(data_size * test_pct)
    test_data = data[-num_test_samples:]
    train_data = data[: data_size - num_test_samples]
    return train_data, test_data


# Split data
train_data, test_data = train_test_split(data)


def get_batch(data):
    """Get a batch of data for appropriate device type"""
    # Get  batch_size of  indices which will be the start of sequence
    # Ensure that indices start before the length of data - block_size to prevent out of bounds error
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate(model):
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros((eval_iters))
        for i in range(eval_iters):
            x, y = get_batch(train_data)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
    """Multiheaded-attention module"""

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


class LinearLayer(nn.Module):
    """FeedForward layer"""

    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Multiheaded-attention followed by FeedForward layer"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.multi_headed_attention = MultiHeadedAttention(self.head_size)
        self.ff = LinearLayer(emb_dim)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # skip connections & layer norm
        x = x + self.multi_headed_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    """Language model with given number of transformer blocks"""

    def __init__(self):
        super().__init__()
        self.head_size = emb_dim // num_heads
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding_table = nn.Embedding(block_size, emb_dim)
        self.transformer_blocks = nn.Sequential(
            *[Block(self.head_size) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape  # batch_size x block_size
        tok_emb = self.token_embedding_table(x)  # batch_size x block_size x emb_dim
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        output = self.transformer_blocks(x)
        output = self.ln_f(output)
        logits = self.linear(output)

        loss = None
        if targets is not None:
            # Logits tranformed to b*t x vocab_size
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @staticmethod
    @torch.no_grad()
    def generate(idx, max_output_size=50):
        """Generate output"""
        for _ in range(max_output_size):
            prompt_trimmed_tnsr = idx[:, -block_size:]
            logits, _ = model(prompt_trimmed_tnsr)
            probs = F.softmax(logits[:, -1, :], dim=1)
            next_tok = torch.multinomial(probs, 1, replacement=True)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx


if __name__ == "__main__":
    model = LanguageModel()
    model = model.to(device)

    # Start with single token (new line character)
    prompt_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Output before training the model
    print("============ Model output before training: ============")
    output_idices = model.generate(prompt_idx, max_output_size=max_output_size)
    # Print generated text
    print(decode(output_idices[0].tolist()))
    print("=======================================================")

    # Initialize optimizer
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr
    )  # , betas=(0.9, 0.99), eps=1e-8)

    # print(generate())
    for i in range(num_iters):
        # x, y = get_batch(train_data, batch_size=batch_size, block_size=block_size)
        x, y = get_batch(train_data)
        logits, loss = model(x, y)
        loss.backward()
        if not (i % 100):
            # print(f'{loss.item()=:.4f}')
            out = evaluate(model)
            print(
                f'step {i}: avg training loss: {out["train"]:.4f} | avg val loss: {out["test"]:.4f}'
            )

        optim.step()
        optim.zero_grad()

    # Prompt the model after training
    output_idices = model.generate(prompt_idx, max_output_size=max_output_size)
    print("============ Model output after training: ============")
    # Print generated text
    print(decode(output_idices[0].tolist()))
