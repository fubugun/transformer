import torch
import torch.nn as nn
import torch.optim as optim
import math
import urllib.request # 下载数据用的
import tarfile# 解压下载的压缩包 把 .tar.bz2 解压成 .csv 数据文件
import csv  # 读取数据文件读取解压后的 sentences.csv 和 links.csv从中提取中文、英文句
from torch.utils.data import Dataset, DataLoader

# ========================
# 超参数
# ========================

d_model = 128
n_heads = 4
d_ff = 256
num_layers = 2

batch_size = 32
lr = 1e-3
epochs = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# 下载 Tatoeba 数据
# ========================

print("下载数据...")

# urllib.request.urlretrieve = 下载文件
urllib.request.urlretrieve(
    # 几百万句中文、英文句子（带编号）
    "https://downloads.tatoeba.org/exports/sentences.tar.bz2",
    "sentences.tar.bz2"
)

urllib.request.urlretrieve(
    # 句子之间的对应关系
    "https://downloads.tatoeba.org/exports/links.tar.bz2",
    "links.tar.bz2"
)

print("解压数据...")

with tarfile.open("sentences.tar.bz2") as tar:
    tar.extractall()

with tarfile.open("links.tar.bz2") as tar:
    tar.extractall()

# ========================
# 构建中英句对
# ========================


#  读取 CSV 文件，把所有【中文、英文句子】存进一个字典里，方便后面查找。
sentences = {}

with open("sentences.csv", encoding="utf-8") as f:

    reader = csv.reader(f, delimiter="\t")

    for row in reader:

        if len(row) != 3:
            continue

        id_, lang, text = row

        if lang in ["eng", "cmn"]:

            sentences[id_] = (lang, text)


pairs = []

with open("links.csv") as f:

    reader = csv.reader(f, delimiter="\t")

    for a, b in reader:

        if a in sentences and b in sentences:

            lang1, text1 = sentences[a]
            lang2, text2 = sentences[b]

            if lang1 == "cmn" and lang2 == "eng":

                pairs.append((text1, text2.lower()))

            elif lang2 == "cmn" and lang1 == "eng":

                pairs.append((text2, text1.lower()))

pairs = pairs[:50000]

print("中英句对数量:", len(pairs))

# ========================
# 构建词典
# ========================
# 把中文、英文文字 → 变成模型能看懂的数字！
# 同时建立 “文字 ↔ 数字” 的词典对照表

SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"

src_vocab = set()
tgt_vocab = set()

for ch, en in pairs:

    src_vocab.update(ch)
    tgt_vocab.update(en.split())

src_vocab = [PAD, SOS, EOS] + sorted(src_vocab)
tgt_vocab = [PAD, SOS, EOS] + sorted(tgt_vocab)

src_word2idx = {w:i for i,w in enumerate(src_vocab)}
tgt_word2idx = {w:i for i,w in enumerate(tgt_vocab)}

tgt_idx2word = {i:w for i,w in enumerate(tgt_vocab)}

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

# ========================
# Transformer
# ========================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            *
            (-math.log(10000.0)/d_model)
        )

        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        self.n_heads = n_heads

        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, k, v, mask=None):

        B, T, C = q.shape

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(B, T, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(B, -1, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(B, -1, self.n_heads, self.d_k).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)

        if mask is not None:

            attn = attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v)

        out = out.transpose(1,2).contiguous()

        out = out.view(B, T, C)

        return self.out(out)



class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):

        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):

        return self.fc2(
            torch.relu(
                self.fc1(x)
            )
        )



class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):

        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_heads)

        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, mask):

        x = self.norm1(
            x + self.attn(x, x, x, mask)
        )

        x = self.norm2(
            x + self.ff(x)
        )

        return x



class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff):

        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)

        self.cross_attn = MultiHeadAttention(d_model, n_heads)

        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)


    def forward(self, x, enc, src_mask, tgt_mask):

        x = self.norm1(
            x + self.self_attn(x, x, x, tgt_mask)
        )

        x = self.norm2(
            x + self.cross_attn(x, enc, enc, src_mask)
        )

        x = self.norm3(
            x + self.ff(x)
        )

        return x



class Transformer(nn.Module):

    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 d_model,
                 n_heads,
                 d_ff,
                 num_layers):

        super().__init__()

        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)

        self.pos = PositionalEncoding(d_model)

        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        self.fc = nn.Linear(d_model, tgt_vocab)


    def encode(self, src, src_mask):

        x = self.pos(self.src_emb(src))

        for layer in self.enc_layers:

            x = layer(x, src_mask)

        return x


    def decode(self, tgt, enc_out, src_mask, tgt_mask):

        x = self.pos(self.tgt_emb(tgt))

        for layer in self.dec_layers:

            x = layer(x, enc_out, src_mask, tgt_mask)

        return self.fc(x)



model = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    n_heads,
    d_ff,
    num_layers
).to(device)

# ========================
# dataset
# ========================

class TransDataset(Dataset):

    def __len__(self):

        return len(pairs)

    def __getitem__(self, idx):

        ch, en = pairs[idx]

        src = [src_word2idx[c] for c in ch]

        tgt = (
            [SOS_IDX]
            +
            [tgt_word2idx[w] for w in en.split()]
            +
            [EOS_IDX]
        )

        return src, tgt



def collate_fn(batch):
    # 统一句子长度 + 补0填充
    srcs, tgts = zip(*batch)

    src = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in srcs],
        padding_value=PAD_IDX,
        batch_first=True
    )

    tgt = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tgts],
        padding_value=PAD_IDX,
        batch_first=True
    )

    return src, tgt



loader = DataLoader(
    # loader是一个迭代器 每一轮都会自动给你：
    # 32 句补好 0 的中文 + 32 句补好 0 的英文

    TransDataset(),    # 数据源
    batch_size=32,     # 一次喂32句
    shuffle=True,      # 打乱顺序
    collate_fn=collate_fn  # 统一句子长度
)

opt = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ========================
# train
# ========================

print("开始训练...")

for epoch in range(epochs):

    total_loss = 0

    for src, tgt in loader:

        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

        tgt_len = tgt.size(1)-1

        tgt_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=device)
        ).bool()

        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        enc_out = model.encode(src, src_mask)

        out = model.decode(
            tgt[:,:-1],
            enc_out,
            src_mask,
            tgt_mask
        )

        loss = criterion(
            out.reshape(-1, tgt_vocab_size),
            tgt[:,1:].reshape(-1)
        )

        # 清空之前的梯度
        #反向传播：算出哪里错了
        #优化器更新模型：改正错误
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print("epoch", epoch+1, "loss", total_loss)

# 把训练好的模型权重保存成文件
torch.save(model.state_dict(), "transformer_translation.pth")

print("训练完成")

# ========================
# 测试
# ========================

def translate(sentence):

    model.eval()

    src = torch.tensor(
        [[src_word2idx.get(c,0) for c in sentence]],
        device=device
    )

    tgt = torch.tensor([[SOS_IDX]], device=device)

    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    enc_out = model.encode(src, src_mask)

    for _ in range(50):

        tgt_len = tgt.size(1)

        tgt_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=device)
        ).bool()

        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        out = model.decode(tgt, enc_out, src_mask, tgt_mask)

        next_token = out.argmax(-1)[:,-1:]

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == EOS_IDX:
            break

    #  数字 → 转回英文
    words = [
        tgt_idx2word[i.item()]
        for i in tgt[0]
        if i not in (PAD_IDX, SOS_IDX, EOS_IDX)
    ]

    return " ".join(words)


tests = [
    "你好",
    "谢谢",
    "我喜欢你",
    "今天天气很好"
]

for t in tests:

    print(t, "->", translate(t))