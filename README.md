# Transformer
---

# 项目结构

13_mini_transformer.py  
基础 Transformer 结构实现（仅验证前向传播）

14_mini_t_generate.py  
基于 Transformer 的自回归生成示例

17_real_translate_1.py
在 Colab 上训练的翻译模型（基础版）

18_real_translate_2.py 
在 Colab 上训练的翻译模型（改进版）

transformer_translation_1.pth
训练好的模型参数

---
论文来源：
Attention Is All You Need (Vaswani et al., 2017)

学习资源：
### nanoGPT 数据示例
https://github.com/karpathy/nanoGPT/tree/master/data

### Transformer 理解视频
论文精读：
https://www.bilibili.com/video/BV1Di4y1c7Zm
10分钟理解 Transformer：
https://www.bilibili.com/video/BV1TZ4y1R7K8

---

本仓库包含从零实现 Transformer 的学习过程，包括：

* Transformer 结构实现
* 自回归生成机制
* 中文 → 英文翻译训练
* 改进训练策略（dropout、label smoothing、beam search）

目标是理解 Encoder–Decoder 架构及其在大语言模型（LLM）中的基础作用。

---

# 项目文件说明

## 1️⃣ 基础结构实现

### `13_mini_transformer.py`

用 PyTorch 实现的精简版 Encoder–Decoder Transformer：

包含：

* sin / cos 位置编码（Positional Encoding）
* Multi-Head Attention
* Feed Forward 网络
* Encoder / Decoder 堆叠结构
* padding mask
* decoder 因果 mask（下三角 mask）

文件末尾使用随机整数构造：src / tgt

执行一次前向传播并输出预测 token id，用于：
验证模型结构正确性
检查张量 shape 是否匹配

__该文件重点理解： Transformer 内部结构如何连接__

---

## 2️⃣ 自回归生成

### `14_mini_t_generate.py`

在同一 Transformer 结构基础上，实现自回归生成（autoregressive generation）。

主要改动：

* MultiHeadAttention 支持 T_q ≠ T_k
  适用于 decoder cross-attention

* 实现 generate_sentence 函数：

生成流程：

从单 token 开始 不断调用：model(src, tgt)
取最后一个位置的 logits：argmax → 得到下一个 token → 拼接到 tgt

__用于理解：语言模型如何逐步生成句子__

由于模型权重是随机初始化,生成的 token id 没有语义，仅用于验证生成流程。

---

## 3️⃣ 翻译任务（基础版）

### `17_real_translate_1.py`
在 Google Colab 上训练的中文 → 英文翻译模型

包含：

* tokenizer
* embedding
* positional encoding
* encoder-decoder attention
* padding mask
* greedy decoding

使用平行语料训练后，可以完成简单翻译。

__用于理解：Transformer 如何应用到实际 NLP 任务__

---

## 4️⃣ 翻译任务（改进版）

### `18_real_translate_2.py`

在 Google Colab 上训练的改进版翻译模型。
在基础版上加入常见训练技巧：

* dropout（防止过拟合）
* label smoothing（提升泛化能力）
* 改进 decoder mask
* beam search 解码
* repetition penalty（减少重复词）

---

# 数据集

使用 Tatoeba Project 中英平行语料：https://www.manythings.org/anki/

文件：cmn.txt

训练使用30000 条句对,适合小规模 Transformer 实验。

仓库中已包含：
训练数据
训练好的模型参数

可直接推理，无需重新训练。

---


# 项目目标

通过本项目可以理解：

* Transformer 结构组成
* 注意力机制如何工作
* decoder 为什么需要 mask
* 自回归生成过程
* tokenizer 的作用
* beam search 如何影响生成结果


