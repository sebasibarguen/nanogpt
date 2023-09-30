from collections import Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

from datasets import load_dataset

# to load python for example
dataset = load_dataset(
    "bigcode/starcoderdata", data_dir="python", split="train", streaming=True
)

SAMPLE_SIZE = 5_000

# Parameters
max_iterations = 10_000
eval_iterations = 100
block_size = 256
batch_size = 32
number_of_embeddings = 384
number_of_heads = 6
number_of_layers = 6
learning_rate = 3e-4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "mps"

small_dataset = dataset.take(SAMPLE_SIZE)

text = ""
vocabulary = Counter()

for batch in small_dataset:
    content = batch["content"]
    text += content


def clean_text(text):
    return text.encode("ascii", errors="ignore").decode()


text = clean_text(text)

vocabulary.update(text)

print("Data Size #", len(text))
characters = sorted(list(vocabulary))
vocabulary_size = len(characters)

print("Vocabulary Size #", vocabulary_size)
print("".join(characters[:100]))

string_to_integer = {c: i for i, c in enumerate(characters)}
integer_to_string = {i: c for i, c in enumerate(characters)}

encode = lambda string: [string_to_integer[c] for c in string]
decode = lambda integers: "".join([integer_to_string[i] for i in integers])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(1337)


def get_batch():
    data = train_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size: int, number_of_embeddings: int) -> None:
        super().__init__()

        self.key = nn.Linear(number_of_embeddings, head_size, bias=False)
        self.query = nn.Linear(number_of_embeddings, head_size, bias=False)
        self.value = nn.Linear(number_of_embeddings, head_size, bias=False)

        self.scale = head_size**-0.5

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-1, -2) * self.scale
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        value = self.value(x)
        out = weights @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int, number_of_heads: int) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size, number_of_embeddings) for _ in range(number_of_heads)]
        )
        self.projection = nn.Linear(head_size * number_of_heads, number_of_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = [head(x) for head in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.projection(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, number_of_embeddings: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(number_of_embeddings, number_of_embeddings * 4),
            nn.ReLU(),
            nn.Linear(4 * number_of_embeddings, number_of_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, number_of_embeddings: int, number_of_heads: int) -> None:
        super().__init__()

        head_size = number_of_embeddings // number_of_heads

        self.self_attention = MultiHeadAttention(number_of_heads, head_size)
        self.feed_forward = FeedForward(number_of_embeddings)
        self.layer_norm1 = nn.LayerNorm(number_of_embeddings)
        self.layer_norm2 = nn.LayerNorm(number_of_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


class GPTModel(nn.Module):
    def __init__(self, vocabulary_size, number_of_embeddings=number_of_embeddings):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, number_of_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, number_of_embeddings)

        self.blocks = nn.Sequential(
            *[
                Block(number_of_embeddings, number_of_heads)
                for _ in range(number_of_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(number_of_embeddings)
        self.lm_head = nn.Linear(number_of_embeddings, vocabulary_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )

        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.layer_norm(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probabilities = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probabilities, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPTModel(vocabulary_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iterations):
    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_iterations == 0 or step == max_iterations - 1:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )


def create_context(string: str):
    return torch.tensor(encode(string), dtype=torch.long, device=device).unsqueeze(0)


# context = torch.zeros((1, 1), dtype=torch.long)
context = create_context('def hello():\n    print("Hello World!")\n\nhello()')
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
