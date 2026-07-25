import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. CAUSAL MULTI-HEAD ATTENTION ---
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super(CausalMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
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

# --- 2. FEED-FORWARD NETWORK ---
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. TRANSFORMER BLOCK ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super(TransformerBlock, self).__init__()
        self.sa = CausalMultiHeadAttention(embed_dim, num_heads, block_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 4. THE FULL MINI-GPT ARCHITECTURE ---
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super(MiniGPT, self).__init__()
        self.block_size = block_size
        
        # Word & Position Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        
        # Stack N Transformer Blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size) for _ in range(num_layers)
        ])
        
        # Final Normalization and Classifier
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        # Get Token & Position Embeddings
        tok_emb = self.token_embedding_table(idx) # [B, T, embed_dim]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [T, embed_dim]
        
        # Add them together
        x = tok_emb + pos_emb # [B, T, embed_dim]
        
        # Pass through Transformer Blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Compute raw output probabilities (logits)
        logits = self.lm_head(x) # [B, T, vocab_size]
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            # Crop current sequence to max block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass to get logits
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# --- TEST THE FULL GPT MODEL ---
if __name__ == "__main__":
    VOCAB_SIZE = 65
    EMBED_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3  # 3 stacked Transformer layers
    BLOCK_SIZE = 32
    
    model = MiniGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Mini-GPT initialized successfully!")
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # Test a dummy forward pass
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, 8)) # Batch size 2, Sequence length 8
    logits, loss = model(dummy_input)
    print("Logits Shape:", logits.shape)