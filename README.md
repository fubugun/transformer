# Mini Transformer

## 项目简介

### `13_mini_transformer.py`

用 PyTorch 实现的**精简版 Encoder–Decoder Transformer**：含 sin/cos 位置编码、多头注意力、前馈、堆叠的 Encoder/Decoder 层，以及共享词嵌入和输出线性层；`make_mask` 里用 `src != 0` 做 padding 掩码、下三角做 decoder 因果掩码。
文件末尾用**随机整数**当 `src`/`tgt` 做一次前向，再 `argmax` 打印预测 id，用来**验证前向与 shape**。

---

### `14_mini_t_generate.py`

在**同一套 Transformer 结构**上，专门为**自回归生成**做了两件事：一是 **`MultiHeadAttention` 支持 `T_q ≠ T_k`**（交叉注意力与不等长 decoder 兼容）；二是提供 **`generate_sentence`**：从单 token 起步，循环调用 `model(src, tgt)`，用**最后一个位置**的 logits 做 `argmax` 再拼到 `tgt` 上。测试段演示随机 `src` 上的生成。权重仍是随机的，输出 id **没有语义**


论文来源：

Attention Is All You Need

---

## 学习资源

### nanoGPT 数据示例

nanoGPT
[https://github.com/karpathy/nanoGPT/tree/master/data](https://github.com/karpathy/nanoGPT/tree/master/data)

### Transformer 理解视频

Transformer 论文逐段精读
[https://www.bilibili.com/video/BV1Di4y1c7Zm](https://www.bilibili.com/video/BV1Di4y1c7Zm)

10分钟速通 Transformer
[https://www.bilibili.com/video/BV1TZ4y1R7K8](https://www.bilibili.com/video/BV1TZ4y1R7K8)

---

## 当前功能

目前已实现：

* Token Embedding
* Positional Encoding
* Causal Mask（防止模型看到未来信息）
* Multi-Head Self Attention
* Feed Forward Network
* 多层 Transformer Block
* 自回归训练目标（预测下一个 token）

模型可以学习序列中 token 的统计规律，并生成简单文本。

---

## 后续计划

### 1. 实现论文中的自然语言翻译任务

在原始论文中，Transformer 被提出用于 **机器翻译任务**：

```text
输入：英文句子
输出：对应的中文句子
```

计划基于当前模型结构，实现一个简化版的翻译模型，以更深入理解 Encoder-Decoder 架构。

---

### 2. 复现 nanoGPT

在 Mini Transformer 的基础上，逐步实现：

* 更完整的 GPT 结构
* 更稳定的训练流程
* 文本生成函数（generate）
* 在小数据集上训练语言模型

复现nanoGPT

---

