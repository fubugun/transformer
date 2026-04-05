# Mini Transformer

本项目从零实现了一个 **最小版本的 Transformer 模型**，用于理解 Transformer 的核心机制，并逐步复现 GPT 类模型的基本功能。

## 项目简介

`Mini_transformer.py` 构建了一个基于 **Transformer** 架构的神经网络，包括：

* **Embedding 层**：将 token 映射为向量表示
* **Self-Attention 机制**：建模序列中不同 token 之间的依赖关系
* **Feed Forward Network (前馈网络)**：对特征进行非线性变换
* **多层 Transformer Block 堆叠**：提升模型表达能力

模型的训练目标是：

> 根据已有的 token 序列，预测每个位置的下一个 token 的概率分布（自回归语言模型）

该过程是 GPT 类模型生成文本能力的核心原理。

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

