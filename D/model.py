import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, pe_type='sinusoidal', dropout=0.1):
        super().__init__()
        self.pe_type = pe_type
        self.dropout = nn.Dropout(dropout)

        if pe_type == 'sinusoidal':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        elif pe_type == 'learnable':
            self.pe = nn.Embedding(max_len, d_model)
            self.position_ids = torch.arange(max_len).unsqueeze(0)
        elif pe_type == 'none':
            pass
        else:
            raise ValueError("pe_type must be 'sinusoidal', 'learnable', or 'none'")

    def forward(self, x):
        if self.pe_type == 'sinusoidal':
            x = x + self.pe[:, :x.size(1), :]
        elif self.pe_type == 'learnable':
            pos_ids = self.position_ids[:, :x.size(1)].to(x.device)
            x = x + self.pe(pos_ids)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.size()
        Q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        self.attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(self.attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.dropout(self.W_o(attn_out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class SublayerConnection(nn.Module):
    """遵循17年论文 Post-LN 结构"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, num_heads, d_ff, max_len, pe_type, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, pe_type, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)