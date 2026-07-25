import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Causal Multi-Head Attention (From Step 2)
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

# 2. Feed-Forward Network (The "Thinking" Layer)
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(FeedForward, self).__init__()
        # Standard GPT architecture expands dimensions 4x in the middle layer
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(), # Smooth activation function used in GPT models
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

# 3. The Full Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super(TransformerBlock, self).__init__()
        self.sa = CausalMultiHeadAttention(embed_dim, num_heads, block_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # We use Pre-LN (LayerNorm before the operation) with Residual Additions (+)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- TEST THE TRANSFORMER BLOCK ---
if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 8
    EMBED_DIM = 64
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    
    block = TransformerBlock(EMBED_DIM, NUM_HEADS, BLOCK_SIZE)
    output = block(dummy_input)
    
    print("Input Tensor Shape: ", dummy_input.shape)
    print("Output Tensor Shape:", output.shape)
    print("\nTransformer Block successfully processed the sequence without altering shape!")