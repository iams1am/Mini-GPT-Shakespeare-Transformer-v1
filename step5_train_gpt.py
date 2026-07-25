import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import urllib.request
import time

# --- HYPERPARAMETERS (Tuned for NVIDIA MX330) ---
BATCH_SIZE = 64      # Number of parallel sequences to process
BLOCK_SIZE = 128     # Maximum context length for attention window
MAX_ITERS = 2000     # Total training iterations
EVAL_INTERVAL = 200  # How often to check loss
LEARNING_RATE = 1e-3 # AdamW learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBED_DIM = 128      # Vector dimension size
NUM_HEADS = 4        # Parallel attention heads (128 // 4 = 32 per head)
NUM_LAYERS = 4       # Stacked Transformer blocks

# --- 1. DATA PREPARATION ---
file_path = 'shakespeare.txt'
if not os.path.exists(file_path):
    print("Downloading shakespeare.txt...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, file_path)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)) # 90% train, 10% validation
train_data = data[:n]
val_data = data[n:]

# Efficient parallel batch loader
def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    # Grab random starting indexes across the dataset
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([dataset[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 2. ARCHITECTURE DEFINITION ---
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.sa = CausalMultiHeadAttention(embed_dim, num_heads, block_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        self.blocks = nn.Sequential(*[
            TransformerBlock(EMBED_DIM, NUM_HEADS, BLOCK_SIZE) for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

# --- 3. TRAINING LOOP ---
model = MiniGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training Mini-GPT on {DEVICE.upper()}...")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 50)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

start_time = time.time()

for iter in range(MAX_ITERS + 1):
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS:
        losses = estimate_loss()
        vram_used = torch.cuda.max_memory_allocated() / 1024**2 if DEVICE == 'cuda' else 0
        print(f"Iter [{iter:4d}/{MAX_ITERS}] | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | VRAM: {vram_used:.1f} MB")

    # Sample batch and calculate gradients
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

total_time = time.time() - start_time
print("-" * 50)
print(f"Training Complete in {total_time:.1f}s!")

# Save the trained Transformer weights
torch.save(model.state_dict(), 'mini_gpt_shakespeare.pth')
print("Saved model checkpoint to mini_gpt_shakespeare.pth")