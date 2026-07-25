import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super(CausalMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # We project Q, K, and V all in one single matrix multiplication for maximum GPU speed
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim) # Output projection
        
        # Create a lower-triangular mask (1s on and below the diagonal, 0s above)
        # register_buffer tells PyTorch this is a fixed mask, not a trainable weight
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim

        # Calculate Q, K, V for all heads at once
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # Reshape for multi-head attention: [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Compute raw Attention Scores (Q @ K^T)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # 2. CAUSAL MASKING: Replace upper-triangle (future tokens) with -infinity
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # 3. Softmax turns -inf into exactly 0.0 probability!
        att = F.softmax(att, dim=-1)

        # 4. Multiply scores by Values
        y = att @ v # Shape: [B, num_heads, T, head_dim]

        # Concatenate all heads back together: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), att

# --- TEST THE CAUSAL MASK ---
if __name__ == "__main__":
    BATCH_SIZE = 1
    SEQ_LEN = 5
    EMBED_DIM = 16
    NUM_HEADS = 2   # 2 attention heads running in parallel
    BLOCK_SIZE = 10 # Maximum context window length
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    
    multihead_layer = CausalMultiHeadAttention(EMBED_DIM, NUM_HEADS, BLOCK_SIZE)
    output, attn_weights = multihead_layer(dummy_input)
    
    print("Multi-Head Output Shape:", output.shape)
    print("\nCausal Attention Heatmap for Head 0:\n")
    # Format floating point numbers to look clean
    torch.set_printoptions(precision=3, sci_mode=False)
    print(attn_weights[0, 0].detach())
