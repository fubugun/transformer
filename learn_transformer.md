

# 1. Transformer为何使用多头注意力机制？（为什么不使用一个头）

**回答**

单头注意力只能学习一种关系模式，而语言中存在多种依赖关系，例如：

* 语法关系（主谓一致）
* 语义关系（同义词）
* 长距离依赖
* 局部依赖

多头注意力将输入映射到多个子空间，使模型能够并行关注不同特征：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(head_1,...,head_h)W^O
$$

其中每个头：

$$
head_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

优点：

* 捕获多种语义关系
* 提升表达能力
* 提高模型稳定性

---

# 2. Transformer为什么Q和K使用不同的权重矩阵生成？

**回答**

Q（Query）表示当前token想寻找的信息
K（Key）表示当前token能提供的信息

映射方式：

$$
Q=XW^Q,\quad K=XW^K
$$

如果使用相同矩阵：

$$
Q=K
$$

则：

$$
QK^T = XX^T
$$

问题：

* 无法区分查询和被查询
* 表达能力下降
* 注意力机制退化

不同矩阵允许模型学习不同的语义空间。

---

# 3. Transformer计算attention为何使用点乘而不是加法？

**回答**

加法注意力：

$$
score=v^T\tanh(W_1Q+W_2K)
$$

点乘注意力：

$$
score=QK^T
$$

复杂度比较：

设向量维度为 d：

| 方法 | 复杂度    |
| -- | ------ |
| 加法 | O(nd²) |
| 点乘 | O(nd)  |

点乘注意力优点：

* 计算更快
* GPU并行效率高
* 实现简单

实验表明两者效果接近，但点乘更高效。

---

# 4. 为什么attention需要scaled（除以 √dk）？

Attention公式：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

设：

q,k 服从均值0方差1分布：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

则：

$$
Var(q \cdot k)=d_k
$$

维度越大：

点积值越大 → softmax进入饱和区 → 梯度变小。

缩放后：

$$
Var\left(\frac{qk}{\sqrt{d_k}}\right)=1
$$

作用：

* 防止softmax饱和
* 稳定梯度
* 加速训练

---

# 5. 在计算attention score时如何对padding做mask？

padding token不应参与计算：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^T + M}{\sqrt{d_k}}\right)V
$$

mask矩阵：

padding位置：

$$
M=-\infty
$$

实现：

```python
scores = Q @ K.transpose(-2,-1)
scores = scores.masked_fill(mask==0, -1e9)
attn = softmax(scores)
```

效果：

padding位置权重接近0。

---

# 6. 为什么多头注意力需要对每个head降维？

设：

$$
d_{model}=512,\quad h=8
$$

每个head维度：

$$
d_k = \frac{d_{model}}{h}
$$

原因：

1. 保持计算量不变

单头：

$$
O(d_{model}^2)
$$

多头：

$$
h \times O\left(\left(\frac{d_{model}}{h}\right)^2\right)
\approx O(d_{model}^2)
$$

2. 防止参数过多

3. 提高泛化能力

---

# 7. Transformer Encoder模块

每层结构：

1. Multi-head self-attention
2. Feed Forward Network

结构：

```
input
 ↓
Multi-head attention
 ↓
Add & LayerNorm
 ↓
Feed Forward
 ↓
Add & LayerNorm
```

特点：

* token之间可相互关注
* 并行计算
* 通常堆叠6层

输出：上下文表示。

---

# 8. 为什么embedding要乘 √d_model？

缩放：

$$
x_{scaled} = x \cdot \sqrt{d_{model}}
$$

原因：

embedding初始化较小：

$$
Var(x)\approx \frac{1}{d_{model}}
$$

位置编码幅值约为1。

不缩放：

位置编码影响过大。

缩放后：

两者量级一致。

---

# 9. Transformer位置编码

公式：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

作用：

提供序列顺序信息。

优点：

* 可外推到长序列
* 不需要训练参数
* 表达相对距离

缺点：

* 固定模式
* 不一定最优

---

# 10. 其他位置编码方法

### learned positional embedding

优点：

灵活

缺点：

无法处理更长序列

---

### RoPE

优点：

* 表达相对位置
* 长文本效果好

---

### ALiBi

优点：

可外推

缺点：

表达能力较弱

---

# 11. Transformer残差结构

公式：

$$
y = x + F(x)
$$

作用：

* 防止梯度消失
* 保留原始信息
* 加速训练

---

# 12. 为什么使用LayerNorm而不是BatchNorm？

LayerNorm：

$$
LN(x)=\frac{x-\mu}{\sigma}
$$

优点：

* 不依赖batch size
* 适合NLP
* 适合变长序列

位置：

每个子层后：

Add & Norm。

---

# 13. BatchNorm技术

公式：

$$
BN(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

优点：

* 收敛更快
* 稳定训练

缺点：

* 依赖batch统计
* 不适合序列任务

---

# 14. Transformer前馈网络FFN

公式：

$$
FFN(x)=\max(0,xW_1+b_1)W_2+b_2
$$

结构：

两层MLP：

$$
d_{model} \rightarrow d_{ff} \rightarrow d_{model}
$$

激活函数：

ReLU 或 GELU

作用：

增加非线性能力。

---

# 15. Encoder和Decoder如何交互？

Decoder包含cross-attention：

$$
Attention(Q_{dec},K_{enc},V_{enc})
$$

Q来自decoder：

K,V来自encoder。

作用：

decoder根据输入序列生成输出。

---

# 16. Decoder自注意力与Encoder区别

Decoder需要mask未来信息：

$$
M_{ij}=
\begin{cases}
0 & j>i \
1 & j\le i
\end{cases}
$$

原因：

生成第i个token时不能看到未来token。

Encoder：

不需要mask。

---

# 17. Transformer并行化体现

并行部分：

* attention计算
* token计算
* 多头计算

训练时：

Encoder和Decoder可并行。

推理时：

Decoder必须逐token生成。

---

# 19. 学习率与Dropout

学习率：

$$
lr=d^{-0.5}\min(step^{-0.5},step\cdot warmup^{-1.5})
$$

Dropout位置：

* attention权重
* embedding
* FFN
* residual连接

测试时：

关闭dropout。

---

# 20. Decoder残差是否造成信息泄露？

不会。

原因：

mask已经作用在attention中：

未来token权重接近0。

残差只作用当前层输入：

不会引入未来信息。

---


## Encoder / Decoder 区别与场景（整理版）

---

### 核心差别（一句话各一句）

| | **Encoder（编码器）** | **Decoder（解码器）** |
|---|------------------------|------------------------|
| **怎么看输入** | **整段一起看**，token 之间**可以互相看**（双向/全连接注意力，视具体设计） | **生成时常用因果注意力**：**只能看已经生成的左边，不能偷看未来** |
| **典型输出** | **每个位置的向量**或**整句向量**（表示、特征） | **下一个 token 的概率分布**，**自回归一段一段往外写** |
| **更像** | **读懂、打分、抽特征** | **续写、对话、生成** |

---

### 功能上各自擅长啥

| **Encoder-only（只要编码器）** | **Decoder-only（只要解码器）** |
|--------------------------------|----------------------------------|
| 文本 / 句子 **分类**（情感、主题、审核） | **文本生成**（续写、对话、写代码） |
| **序列标注**（NER、分词：每个字一个标签） | **指令跟随**（提示词 → 补全） |
| **语义向量、相似度、检索**（搜文档、RAG 里找片段） | **单模型统一成「接着写」** 做多种任务 |
| **句子对**关系（是否重复、蕴含、匹配） | 大规模 **下一 token 预训练** 最顺 |

---

### 常见「合体」：Encoder + Decoder

| **场景** | **为啥要两个** |
|----------|----------------|
| **机器翻译** | 源语言 **Encoder 读懂** → **Decoder 生成**目标语言 |
| **摘要（seq2seq 式）** | **Encoder 读长文** → **Decoder 写短摘要** |
| **某些语音/图像到文本** | 一侧编码输入模态，一侧解码成文本 |



---

### 和聊天大模型（ChatGPT / DeepSeek 等）的关系

- **主生成模型**多是 **Decoder-only**（续写友好、工程成熟）。  
- **产品里**还可能加：**Encoder 式检索 / 向量化**、路由、工具等，**不等于**「整个系统只有 Decoder」。

---

### 对照

1. **Encoder**：偏 **理解与表示**（分类、检索、向量）。  
2. **Decoder**：偏 **生成**（按顺序吐字）。  
3. **Encoder–Decoder**：偏 **「读 A 写 B」**（翻译、seq2seq 摘要等）。