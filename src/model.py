import torch
import torch.nn as nn
import math

# ---- RMSNorm ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.weight

# ---- Rotary Position Embeddings (RoPE) ----
def get_rope_cache(seq_len, dim, device):
    theta = 10000 ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", pos, theta)  # (seq_len, dim/2)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim/2)
    return cos.to(device), sin.to(device)

def apply_rotary_emb(q, k, cos, sin):
    # q, k: (B, H, T, D)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot

# ---- Grouped Query Attention (GQA) ----
class GQAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim * num_kv_heads // num_heads)
        self.v_proj = nn.Linear(dim, dim * num_kv_heads // num_heads)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, Hq, T, D)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand k/v for grouped-query attention
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        q, k = apply_rotary_emb(q, k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, D)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ---- Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = GQAttention(config.hidden_size, config.num_heads, config.num_kv_heads)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

# ---- Full MiniHealthLM Model ----
class MiniHealthLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids):
        B, T = input_ids.size()
        device = input_ids.device
        head_dim = self.config.hidden_size // self.config.num_heads
        cos, sin = get_rope_cache(T, head_dim, device)

        targets = input_ids  # ← Save the token IDs as targets
        x = self.token_embed(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
