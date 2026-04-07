import torch
import torch.nn as nn
import torch.optim as optim
import math
import urllib.request
import tarfile
import csv
from torch.utils.data import Dataset, DataLoader



d_model = 128
n_heads = 4
d_ff = 256
num_layers = 2

dropout = 0.1
label_smoothing = 0.05

batch_size = 32
lr = 1e-3
epochs = 40

beam_size = 3
max_len = 20
max_sentence_len = 12   # 过滤过长句子

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# 下载数据
# ========================

print("下载数据...")

urllib.request.urlretrieve(
    "https://downloads.tatoeba.org/exports/sentences.tar.bz2",
    "sentences.tar.bz2"
)

urllib.request.urlretrieve(
    "https://downloads.tatoeba.org/exports/links.tar.bz2",
    "links.tar.bz2"
)

print("解压数据...")

with tarfile.open("sentences.tar.bz2") as tar:
    tar.extractall(filter="data")

with tarfile.open("links.tar.bz2") as tar:
    tar.extractall(filter="data")

# ========================
# 构建中英句对
# ========================

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

# 限制长度
pairs = [
    (zh, en)
    for zh, en in pairs
    if len(zh) <= max_sentence_len
    and len(en.split()) <= max_sentence_len
]

pairs = pairs[:30000]

print("句对数量:", len(pairs))

# ========================
# 词典
# ========================

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

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# ========================
# mask
# ========================

def create_src_mask(src):

    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt):

    pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)

    T = tgt.size(1)

    causal_mask = torch.tril(
        torch.ones(T, T, device=tgt.device)
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    return pad_mask & causal_mask

# ========================
# 模型
# ========================

class PositionalEncoding(nn.Module):

    def __init__(self):

        super().__init__()

        pe = torch.zeros(5000, d_model)

        pos = torch.arange(0,5000).unsqueeze(1)

        div = torch.exp(

            torch.arange(0,d_model,2)

            *

            (-math.log(10000.0)/d_model)

        )

        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,x):

        return x + self.pe[:,:x.size(1)]


class MultiHeadAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model//n_heads

        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)

        self.out = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):

        B,T,C = q.shape

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = q.view(B,T,self.n_heads,self.d_k).transpose(1,2)
        k = k.view(B,-1,self.n_heads,self.d_k).transpose(1,2)
        v = v.view(B,-1,self.n_heads,self.d_k).transpose(1,2)

        attn = (q@k.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:

            attn = attn.masked_fill(mask==0,-1e9)

        attn = torch.softmax(attn,-1)

        attn = self.dropout(attn)

        out = attn@v

        out = out.transpose(1,2).contiguous()

        out = out.view(B,T,C)

        return self.out(out)


class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(d_model,d_ff),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(d_ff,d_model)

        )

    def forward(self,x):

        return self.net(x)


class EncoderLayer(nn.Module):

    def __init__(self):

        super().__init__()

        self.attn = MultiHeadAttention()

        self.ff = FeedForward()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):

        x = self.norm1(

            x + self.dropout(

                self.attn(x,x,x,mask)
            )
        )

        x = self.norm2(

            x + self.dropout(

                self.ff(x)
            )
        )

        return x


class DecoderLayer(nn.Module):

    def __init__(self):

        super().__init__()

        self.self_attn = MultiHeadAttention()

        self.cross_attn = MultiHeadAttention()

        self.ff = FeedForward()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc,src_mask,tgt_mask):

        x = self.norm1(

            x + self.dropout(

                self.self_attn(x,x,x,tgt_mask)
            )
        )

        x = self.norm2(

            x + self.dropout(

                self.cross_attn(x,enc,enc,src_mask)
            )
        )

        x = self.norm3(

            x + self.dropout(

                self.ff(x)
            )
        )

        return x


class Transformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.src_emb = nn.Embedding(src_vocab_size,d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size,d_model)

        self.pos = PositionalEncoding()

        self.enc = nn.ModuleList(

            [EncoderLayer() for _ in range(num_layers)]
        )

        self.dec = nn.ModuleList(

            [DecoderLayer() for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model,tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self,src,mask):

        x = self.dropout(

            self.pos(

                self.src_emb(src)
            )
        )

        for layer in self.enc:

            x = layer(x,mask)

        return x


    def decode(self,tgt,enc,src_mask,tgt_mask):

        x = self.dropout(

            self.pos(

                self.tgt_emb(tgt)
            )
        )

        for layer in self.dec:

            x = layer(x,enc,src_mask,tgt_mask)

        return self.fc(x)


model = Transformer().to(device)

# ========================
# dataset
# ========================

class TransDataset(Dataset):

    def __len__(self):

        return len(pairs)

    def __getitem__(self,idx):

        ch,en = pairs[idx]

        src = [src_word2idx[c] for c in ch]

        tgt = (

            [SOS_IDX]

            +

            [tgt_word2idx[w] for w in en.split()]

            +

            [EOS_IDX]
        )

        return src,tgt


def collate_fn(batch):

    srcs,tgts = zip(*batch)

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

    return src,tgt


loader = DataLoader(

    TransDataset(),

    batch_size=batch_size,

    shuffle=True,

    collate_fn=collate_fn
)

# ========================
# label smoothing
# ========================

class LabelSmoothingLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self,pred,target):

        pred = torch.log_softmax(pred,-1)

        true_dist = torch.zeros_like(pred)

        true_dist.fill_(

            label_smoothing/(tgt_vocab_size-1)
        )

        true_dist.scatter_(

            1,

            target.unsqueeze(1),

            1-label_smoothing
        )

        true_dist[target==PAD_IDX] = 0

        return self.kl(pred,true_dist)


criterion = LabelSmoothingLoss()

opt = optim.Adam(model.parameters(),lr=lr)

# ========================
# train
# ========================

print("开始训练...")

for epoch in range(epochs):

    total_loss = 0

    for src,tgt in loader:

        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = create_src_mask(src)

        tgt_input = tgt[:,:-1]

        tgt_mask = create_tgt_mask(tgt_input)

        enc_out = model.encode(src,src_mask)

        out = model.decode(

            tgt_input,

            enc_out,

            src_mask,

            tgt_mask
        )

        loss = criterion(

            out.reshape(-1,tgt_vocab_size),

            tgt[:,1:].reshape(-1)
        )

        opt.zero_grad()

        loss.backward()

        opt.step()

        total_loss += loss.item()

    print("epoch",epoch+1,"loss",total_loss)

print("训练完成")

# ========================
# beam search
# ========================

def translate(sentence):

    model.eval()

    src = torch.tensor(

        [[src_word2idx.get(c,0) for c in sentence]],

        device=device
    )

    src_mask = create_src_mask(src)

    enc_out = model.encode(src,src_mask)

    beams = [(

        torch.tensor([[SOS_IDX]],device=device),

        0
    )]

    for _ in range(max_len):

        new_beams = []

        for seq,score in beams:

            tgt_mask = create_tgt_mask(seq)

            out = model.decode(

                seq,

                enc_out,

                src_mask,

                tgt_mask
            )

            prob = torch.log_softmax(out[:,-1],-1)

            topk = torch.topk(prob,beam_size)

            for i in range(beam_size):

                next_tok = topk.indices[0,i].view(1,1)

                new_seq = torch.cat([seq,next_tok],1)

                new_score = score + topk.values[0,i].item()

                new_beams.append((new_seq,new_score))

        beams = sorted(

            new_beams,

            key=lambda x:x[1],

            reverse=True

        )[:beam_size]

    best = beams[0][0]

    words = [

        tgt_idx2word[i.item()]

        for i in best[0]

        if i not in (PAD_IDX,SOS_IDX,EOS_IDX)
    ]

    return " ".join(words)

# ========================
# test
# ========================

tests = [

    "你好",

    "谢谢",

    "我喜欢你",

    "今天天气很好"
]

for t in tests:

    print(t,"->",translate(t))