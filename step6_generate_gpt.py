import torch
import torch.nn as nn
import torch.nn.functional as F

# --- HYPERPARAMETERS (Must match training) ---
BLOCK_SIZE = 128
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load vocabulary mappings
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# --- ARCHITECTURE RECONSTRUCTION ---
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
        return logits

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            # Crop to max block size
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)
            # Take last position and apply temperature
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- GENERATION EXECUTION ---
print(f"Loading weights onto {DEVICE.upper()}...")
model = MiniGPT().to(DEVICE)
model.load_state_dict(torch.load('mini_gpt_shakespeare.pth', map_location=DEVICE, weights_only=True))
model.eval()

# Prompt seed (Try changing this to anything!)
prompt = "KING RICHARD:\n"
context = torch.tensor([[char_to_int[c] for c in prompt]], dtype=torch.long, device=DEVICE)

print("\nGenerating Shakespeare using Mini-GPT...\n")
print("-" * 50)
generated_indices = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
generated_text = "".join([int_to_char[i] for i in generated_indices])
print(generated_text)
print("-" * 50)