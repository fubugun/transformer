import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # 自注意力 Q/K/V 等长；交叉注意力 decoder 的 Q 与 encoder 的 K/V 长度可不同
        B, T_q, C = Q.shape
        T_k = K.size(1)
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Q = Q.view(B, T_q, self.heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        output = attn @ V
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T_q, C)
        return self.out(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model = 128,
        heads = 4,
        d_ff = 256,
        layers = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff) for _ in range(layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff) for _ in range(layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def make_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_mask(src, tgt)
        src = self.pos(self.embed(src))
        tgt = self.pos(self.embed(tgt))
        for layer in self.encoder:
            src = layer(src, src_mask)
        enc_out = src
        for layer in self.decoder:
            tgt = layer(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(tgt)

def generate_sentence(model, src, max_len=5):
    
    device = next(model.parameters()).device
    src = src[:1].to(device)
    tgt = torch.zeros(1, 1, dtype=torch.long, device=device)
    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out.argmax(-1)[:, -1:]
        tgt = torch.cat([tgt, next_token], dim=1)
    return tgt.squeeze()

model = Transformer(vocab_size=1000)
src = torch.randint(0, 1000, (1, 5))
tgt = torch.randint(0, 1000, (1, 5))
out = model(src, tgt)

# 生成句子
generated = generate_sentence(model, src)
print("生成句子：", generated)