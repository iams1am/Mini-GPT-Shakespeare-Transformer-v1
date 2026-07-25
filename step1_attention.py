import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_size):
        super(SingleHeadAttention, self).__init__()
        # Linear projections to turn embeddings into Queries, Keys, and Values
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self, x):
        # B = Batch Size, T = Sequence Length (Time), C = Embedding Channels
        B, T, C = x.shape
        
        # 1. Project inputs into Query, Key, and Value spaces
        k = self.key(x)   # Shape: [B, T, head_size]
        q = self.query(x) # Shape: [B, T, head_size]
        v = self.value(x) # Shape: [B, T, head_size]

        # 2. Compute Attention Scores ("Affinity" between words)
        # Multiply Query by transposed Key: [B, T, head_size] @ [B, head_size, T] -> [B, T, T]
        # We divide by (head_size ** 0.5) to scale the values and keep training stable
        head_size = k.shape[-1]
        weights = q @ k.transpose(-2, -1) * (head_size ** -0.5)

        # 3. Apply Softmax to convert raw scores into percentages (0% to 100%)
        weights = F.softmax(weights, dim=-1)

        # 4. Multiply attention weights by Values
        # [B, T, T] @ [B, T, head_size] -> [B, T, head_size]
        out = weights @ v
        
        return out, weights

# --- TEST IT OUT ---
if __name__ == "__main__":
    BATCH_SIZE = 2
    SEQ_LEN = 5      # 5 words in a sentence
    EMBED_DIM = 16   # Vector size per word
    HEAD_SIZE = 8    # Dimension of Query/Key/Value
    
    # Create fake data representing a batch of 2 sentences, each 5 words long
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    
    attention_layer = SingleHeadAttention(EMBED_DIM, HEAD_SIZE)
    output, attn_weights = attention_layer(dummy_input)
    
    print("Input Shape:", dummy_input.shape)
    print("Attention Weights Heatmap Shape:", attn_weights.shape)
    print("Output Shape:", output.shape)
    print("\nAttention Heatmap for Batch 0 (How word rows attend to word columns):\n")
    print(attn_weights[0].detach())
