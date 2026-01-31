Transformer 之所以能取代 RNN/LSTM ，最大的原因就是它**抛弃了“时间步（Time Step）”的循环依赖**，实现了**训练时的并行化**。

### 1. 痛点：为什么 RNN 无法并行？

在 RNN（或 LSTM）中，网络的计算是串行的。

假设我们要训练一句话：“我 爱 机器人”。

- **t=1:** 输入“我”，计算隐藏层 $h_1$，预测“爱”。
- **t=2:** 输入“爱”，**必须拿到 $h_1$**，才能计算 $h_2$，预测“机器人”。
- **t=3:** ...

**瓶颈：** 你如果不把第 1 步算完，第 2 步就根本没法开始。无论你有多少个 GPU，你也只能等着前一步算完。这就像接力跑，必须等上一棒交接。

### 2. Transformer 的并行训练

Transformer 的结构里没有“循环”。它的输入不是一个一个进来的，而是**整个句子的矩阵（Matrix）一次性“拍”进网络里的**。

#### 核心机制 A：Encoder 的并行（天然并行）

Encoder 负责“理解”句子。

- **输入：** 形状为 `(Batch_Size, Sequence_Length, Embedding_Dim)` 的大矩阵。
- **处理：** Self-Attention 机制是通过**矩阵乘法**实现的。
  - 比如计算 Attention Score，是 $Q \times K^T$。
  - 这意味这，单词“我”去注意“机器人”，和单词“机器人”去注意“我”，这两个计算过程是**同时发生**在矩阵运算里的。
  - GPU 最擅长做这种大规模矩阵乘法。

**结论：** Encoder 在处理输入的 100 个单词时，是 100 个单词同时处理，不需要排队。

#### 核心机制 B：Decoder 的并行（Teacher Forcing + Mask）

这是最难理解的部分。Decoder 负责生成句子。

你可能会问：“生成句子不是必须一个一个崩出来吗？先生成‘爱’，才能生成‘机器人’啊？”

推理时（Inference）： 是的，必须串行。

训练时（Training）： 不是！我们可以并行。

这里用到了 **Teacher Forcing** 技术。

**场景：**

- **输入（Input）：** `<sos>` 我 爱 机器人
- **目标（Target）：** 我 爱 机器人 `<eos>`

在训练时，我们**已经知道**正确答案了（上帝视角）。我们不需要等模型自己瞎猜出“爱”字再喂给下一步。我们可以直接把**正确的标准答案**（Ground Truth）喂给 Decoder。

**训练过程如下（一步到位）：**

1. **输入准备：** 我们把整个句子 `<sos> 我 爱 机器人` 作为输入矩阵 $X$ 一次性喂给 Decoder。
2. **并行预测：** 模型在**同一时刻**输出了 4 个预测结果。
   - 位置 1 的输出：预测 `<sos>` 后面是啥？（目标：我）
   - 位置 2 的输出：预测 `我` 后面是啥？（目标：爱）
   - 位置 3 的输出：预测 `爱` 后面是啥？（目标：机器人）
   - 位置 4 的输出：预测 `机器人` 后面是啥？（目标：`<eos>`）
3. **计算 Loss：** 这 4 个预测位置的 Loss 是同时计算出来的，然后取平均。

### 3. 既然并行输入了，怎么防止“作弊”？（Causal Masking）

如果把 `<sos> 我 爱 机器人` 一次性全喂进去，预测“我”后面的字时，模型利用 Self-Attention 可能会偷看到后面的“爱”字。这在训练时属于作弊（Leakage），会导致测试时（没有后续字）模型失效。

为了解决这个问题，Transformer 引入了 **Masked Self-Attention（掩码注意力）**。

这是一个**上三角矩阵**（Upper Triangular Mask），里面的值为 $-\infty$（负无穷）。

| **位置**          | **可以看到 T1** | **可以看到 T2** | **可以看到 T3** | **可以看到 T4** |     |
| --------------- | ----------- | ----------- | ----------- | ----------- | --- |
| **预测 T1 (我)**   | ✅           | 🚫Mask      | 🚫Mask      | 🚫Mask      |     |
| **预测 T2 (爱)**   | ✅           | ✅           | 🚫Mask      | 🚫Mask      |     |
| **预测 T3 (机器)**  | ✅           | ✅           | ✅           | 🚫Mask      |     |
| **预测 T4 (EOS)** | ✅           | ✅           | ✅           | ✅           |     |

**在 Softmax 之前加上这个 Mask：**

- 当模型在位置 1 计算 Attention 时，**位置 2, 3, 4 的分数会被加上 $-\infty$，此为计算attention bias 矩阵**
- 由于
$$Attention(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \mathbf{Bias}\right)V$$Softmax 之后概率变为 0。
- **物理意义：** 虽然整个句子都在矩阵里，但我强行把这一行后面的数据“抹黑”，让模型看不见未来。
- 这样在乘上V矩阵以后，时序上较先的token对应的行

### 4. 总结：并行训练流程图

1. **数据准备：** 拿到一整个 Batch 的句子。
2. **Embedding：** 把所有单词同时转成向量。
3. **Add Positional Encoding：** 把位置信息加进去（因为没有时间步了，得手动加位置标记）。
4. **Encoder：** 全并行运算，通过矩阵乘法提取特征。
5. **Decoder 输入：** 放入整个目标句子（Shifted Right，即右移一位）。
6. **Masking：** 盖住未来，防止作弊。
7. **Decoder 输出：** **一次性**算出所有位置的预测概率分布。
8. **Loss：** **一次性**拿所有位置的预测结果和真实标签做 CrossEntropy，算出总 Loss。

这就是为什么 Transformer 训练极其快，能吃满 GPU 显存，而 RNN 只能在那慢吞吞地磨洋工。

这是一个非常棒的问题！搞清楚这点，你对 Transformer 的理解就超越了 80% 的初学者。



---

**推理时：** 是**“摸着石头过河”**，上一步生成的字是下一步的输入，**串行**计算（Auto-regressive）。

### 2. 推理模式 (Inference Mode)

**核心策略：Auto-regressive（自回归）**

在推理（应用）时，我们没有答案。模型必须依赖自己上一步的输出来决定这一步走哪里。这是一个**死循环**，直到模型生成结束符 `<EOS>`。

- **Step 1:**
  - **输入：** 只有 `<SOS>`。
  - **处理：** Decoder 运算，输出概率最高的词 -> **“我”**。
- **Step 2:**
  - **输入：** 把上一步生成的“我”拼接到输入序列中 -> `<SOS> 我`。
  - **处理：** Decoder 重新运算整个序列（或者利用 KV Cache 优化），输出下一个词 -> **“爱”**。
- **Step 3:**
  - **输入：** 再次拼接 -> `<SOS> 我 爱`。
  - **处理：** Decoder 运算，输出 -> **“机器人”**。
- **Step 4:**
  - **输入：** `<SOS> 我 爱 机器人`。
  - **处理：** Decoder 运算，输出 -> **`<EOS>`**。
- **终止：** 遇到 `<EOS>` 停止。

## 推理时，在预测第k个token时，也是基于当前1~k位的输入，一次性预测1~k位的输出，其中1~k-1位输出是基于1~k-1位输入做出的，和之前的重复，我们只取最后一位输出加到生成的句子末尾。



------

**训练时和推理时，输入和输出的矩阵形状（Shape）是不一样的。**

但是，**模型的参数（权重矩阵）是不变的**。Transformer 之所以能处理不同形状的输入输出，是因为它在设计上对“序列长度（Sequence Length）”这个维度是**“弹性”**的。

### 1. 维度对比：训练 vs 推理

假设：

- **B (Batch Size)** = 1（为了简单，假设只有一句话）
- **L (Sequence Length)** = 句子长度
- **V (Vocab Size)** = 词表大小（比如 10000 个词）
- **D (Hidden Dim)** = 隐藏层维度（比如 512）

#### A. 训练时 (Training)

训练是**并行**的，输入是固定长度（通常是经过 Padding 补齐的最大长度，比如 10）。

- **输入形状:** `(1, 10, D)` —— 包含 `<sos> 我 爱 机 器 人 <pad>...`
- **模型处理:** 每一层都保持这个长度 `10`。
- **输出形状:** `(1, 10, V)`
- **含义:** 模型同时吐出了 10 个位置的预测结果。
  - 第 1 个位置预测 `<sos>` 的下一个词。
  - 第 2 个位置预测 `我` 的下一个词。
  - ...
- **结果:** 我们计算 Loss 时，会用到这整个 `(1, 10, V)` 的大矩阵。

#### B. 推理时 (Inference)

推理是**串行**的，序列长度 $L$ 是**动态增长**的。

- **第 1 步:**
  - 输入: `<sos>`
  - 输入形状: `(1, 1, D)`
  - 输出形状: `(1, 1, V)` $\rightarrow$ 取最后一个词，得到“我”
- **第 2 步 (Naive 模式):**
  - 输入: `<sos> 我`
  - 输入形状: `(1, 2, D)`
  - 输出形状: `(1, 2, V)`
  - **关键点:** 这里输出了 2 个结果！
    1. `<sos>` 下一位的预测（是“我”，这个我们早就知道了，**废弃**）。
    2. `我` 下一位的预测（是“爱”，这是我们要的，**保留**）。
  - 我们只取矩阵的**最后一行**（Last Token）。
- **第 3 步:**
  - 输入: `<sos> 我 爱`
  - 输入形状: `(1, 3, D)`
  - 输出形状: `(1, 3, V)` $\rightarrow$ 废弃前两个，只取最后一个。

### 2. 为什么模型不会报错？（全连接层的特性）

你可能会担心：*“最后那个全连接层（Linear Layer），不是需要固定的输入维度吗？”*

PyTorch 中的 `nn.Linear(in_features, out_features)` 实际上是非常灵活的。

- 它只关心**最后一个维度**是否匹配。
- 它不管前面有多少个维度（Batch 维度、Time 维度）。

假设你的输出层定义为 `nn.Linear(512, 10000)`。

- **训练时：** 输入是 `(1, 10, 512)`。Linear 层会把前两个维度 `(1, 10)` 看作 $1 \times 10 = 10$ 个独立的样本，分别进行矩阵乘法。输出变成 `(1, 10, 10000)`。
- **推理第 3 步：** 输入是 `(1, 3, 512)`。Linear 层把它们看作 $1 \times 3 = 3$ 个样本。输出变成 `(1, 3, 10000)`。

**结论：** 只要隐藏层维度 `512` 对得上，序列长度是 1 还是 100，Linear 层都不在乎。这就是 Transformer 能够处理变长输入的数学基础。



---

“Right Shifted”（右移一位）是 Transformer **训练数据预处理**中一个非常关键的步骤，特指**Decoder 的输入（Input）**和**标签（Label/Target）**之间的错位关系。

简单来说：**Decoder 的输入，就是把正确答案整体向右挪了一格，并在最前面塞进去一个开始符 `<SOS>`。**

### 1. 为什么要右移？（核心逻辑）

我们训练模型的目的是：**让它根据“上一个字”，预测“下一个字”。**

- 当我们想让模型预测第 2 个字时，我们必须把第 1 个字喂给它。
- 当我们想让模型预测第 3 个字时，我们必须把第 2 个字喂给它。

这就形成了一种**错位**的对应关系。

### 2. 图解实例

假设我们要训练一句话：“我 爱 机器人”。

加上开始符 <SOS> 和结束符 <EOS>，完整的序列是：

['<SOS>', '我', '爱', '机器人', '<EOS>']

在训练时，我们把这个序列拆成两半：

| **时间步** | **Decoder 输入 (Input)(Right Shifted)** | **真实标签 (Target/Label)(期望模型输出的)** | **逻辑含义**                     |
| ---------- | --------------------------------------- | ------------------------------------------- | -------------------------------- |
| 1          | **`<SOS>`**                             | **我**                                      | 看到“开始”，预测第一个字是“我”   |
| 2          | **我**                                  | **爱**                                      | 看到“我”，预测下一个字是“爱”     |
| 3          | **爱**                                  | **机器人**                                  | 看到“爱”，预测下一个字是“机器人” |
| 4          | **机器人**                              | **`<EOS>`**                                 | 看到“机器人”，预测这句话讲完了   |

仔细看上面的表：

- **输入列**：`['<SOS>', '我', '爱', '机器人']`
- **标签列**：`['我', '爱', '机器人', '<EOS>']`

你会发现，**输入列**实际上就是把**标签列**整体**向右移动了一位**，然后在空出来的最左边补了一个 `<SOS>`。

这就是 **"Right Shifted"** 的由来。

### 3. 代码里是怎么做的？

在 PyTorch 的具体实现中，这通常通过切片（Slicing）操作来实现。

假设 `batch` 里的数据是完整的 `[<SOS>, A, B, C, <EOS>]`：

Python

```
# 假设 full_sentence = [<SOS>, 我, 爱, 机器人, <EOS>]

# 1. Decoder 的输入 (去掉最后一个 <EOS>)
decoder_input = full_sentence[:-1] 
# 结果: [<SOS>, 我, 爱, 机器人] -> 这就是 Right Shifted

# 2. 真实的标签 (去掉第一个 <SOS>)
target_label = full_sentence[1:]
# 结果: [我, 爱, 机器人, <EOS>]
```

### 4. 总结

“Right Shifted” 并不是什么复杂的数学变换，它只是为了构造 **Supervised Learning（监督学习）** 的数据对：

- **Question:** 你看到了什么？（当前序列，以 `<SOS>` 开头）
- **Answer:** 下一位应该是什么？（目标序列，以 `<EOS>` 结尾）

这种错位构造，配合 **Causal Mask**，让 Transformer 能够在一次并行计算中，学会所有位置的“预测下一个词”的任务。

------

# 知乎

**Transformer为何使用[多头注意力机制](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=多头注意力机制&zhida_source=entity)？**（为什么不使用一个头）

- 多头保证了transformer可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。可以类比[CNN](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=CNN&zhida_source=entity)中同时使用**多个滤波器**的作用，直观上讲，多头的注意力**有助于网络捕捉到更丰富的特征/信息。**
- 参考：https://www.zhihu.com/question/341222779

**Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？** （注意和第一个问题的区别）

- 使用Q/K/V不相同可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。
- 同时，由softmax函数的性质决定，实质做的是一个soft版本的arg max操作，得到的向量接近一个one-hot向量（接近程度根据这组数的数量级有所不同）。如果令Q=K，那么得到的模型大概率会得到一个类似单位矩阵的attention矩阵，**这样[self-attention](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=self-attention&zhida_source=entity)就退化成一个point-wise线性映射**。这样至少是违反了设计的初衷。
- 参考：https://www.zhihu.com/question/319339652

**Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？**

- K和Q的点乘是为了得到一个attention score 矩阵，用来对V进行提纯。K和Q使用了不同的W_k, W_Q来计算，可以理解为是在不同空间上的投影。正因为 有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。
- 为了计算更快。矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算attention的时候相当于一个隐层，整体计算量和点积相似。在效果上来说，从实验分析，两者的效果和dk相关，dk越大，加法的效果越显著。

**为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根）**，并使用公式推导进行讲解

- 这取决于softmax函数的特性，如果softmax内计算的数数量级太大，会输出近似one-hot编码的形式，导致梯度消失的问题，所以需要scale
- 那么至于为什么需要用维度开根号，假设向量q，k满足各分量独立同分布，均值为0，方差为1，那么qk点积均值为0，方差为dk，从统计学计算，若果让qk点积的方差控制在1，需要将其除以dk的平方根，是的softmax更加平滑
- 参考：https://www.zhihu.com/question/339723385/answer/782509914

**在计算attention score的时候如何对padding做mask操作？**

- padding位置置为负无穷(一般来说-1000就可以)，再对attention score进行相加。对于这一点，涉及到batch_size之类的，具体的大家可以看一下抱抱脸实现的源代码，位置在这里：[https://github.com/huggingface/transformers/blob/aa6a29bc25b663e1311c5c4fb96b004cf8a6d2b6/src/transformers/modeling_bert.py#L720](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers/blob/aa6a29bc25b663e1311c5c4fb96b004cf8a6d2b6/src/transformers/modeling_bert.py%23L720)

**为什么在进行多头注意力的时候需要对每个head进行降维？**（可以参考上面一个问题）

- 将原有的**高维空间转化为多个低维空间**并再最后进行拼接，形成同样维度的输出，借此丰富特性信息
  - 基本结构：Embedding + Position Embedding，Self-Attention，Add + LN，FN，Add + LN

**为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？**

- embedding matrix的初始化方式是xavier init，这种方式的方差是1/embedding size，因此乘以embedding size的开方使得embedding matrix的方差是1，在这个scale下可能更有利于embedding matrix的收敛。

**简单介绍一下Transformer的位置编码？有什么意义和优缺点？**

- 因为self-attention是位置无关的，无论句子的顺序是什么样的，通过self-attention计算的token的hidden embedding都是一样的，这显然不符合人类的思维。因此要有一个办法能够在模型中表达出一个token的位置信息，transformer使用了固定的positional encoding来表示token在句子中的绝对位置信息。
- [一文读懂Transformer模型的位置编码](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/QxaZTVOUrzKfO7B78EM5Uw)
- [浅谈Transformer模型中的位置表示](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/vXYJKF9AViKnd0tbuhMWgQ)
- [Transformer改进之相对位置编码RPE](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/NPM3w7sIYVLuMYxQ_R6PrA)
- [如何优雅地编码文本中的位置信息？三种positioanl encoding方法简述](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/ENpXBYQ4hfdTLSXBIoF00Q)
- [相对位置编码一)Relative Position Representatitons RPR - Transformer](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/shiyublog/p/11185625.html)
- [相对位置编码(二) Relative Positional Encodings - Transformer-XL](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/shiyublog/p/11236212.html)

**你还了解哪些关于位置编码的技术，各自的优缺点是什么？**（参考上一题）

- 相对位置编码（RPE）1.在计算attention score和weighted value时各加入一个可训练的表示相对位置的参数。2.在生成多头注意力时，把对key来说将绝对位置转换为相对query的位置3.复数域函数，已知一个词在某个位置的词向量表示，可以计算出它在任何位置的词向量表示。前两个方法是词向量+位置编码，属于亡羊补牢，复数域是生成词向量的时候即生成对应的位置信息。

**简单讲一下Transformer中的残差结构以及意义。**

- 就是[ResNet](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=ResNet&zhida_source=entity)的优点，解决梯度消失

**为什么transformer块使用[LayerNorm](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=LayerNorm&zhida_source=entity)而不是[BatchNorm](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=BatchNorm&zhida_source=entity)？LayerNorm 在Transformer的位置是哪里？**

- LN：针对每个样本序列进行Norm，没有样本间的依赖。对一个序列的不同特征维度进行Norm
- CV使用BN是认为channel维度的信息对cv方面有重要意义，如果对channel维度也归一化会造成不同通道信息一定的损失。而同理nlp领域认为句子长度不一致，并且各个batch的信息没什么关系，因此只考虑句子内信息的归一化，也就是LN。

**简答讲一下BatchNorm技术，以及它的优缺点。**

- 优点：
  - 第一个就是可以解决内部协变量偏移，简单来说训练过程中，各层分布不同，增大了学习难度，BN缓解了这个问题。当然后来也有论文证明BN有作用和这个没关系，而是可以使**损失平面更加的平滑**，从而加快的收敛速度。
  - 第二个优点就是缓解了**梯度饱和问题**（如果使用sigmoid激活函数的话），加快收敛。
- 缺点：
  - 第一个，batch_size较小的时候，效果差。这一点很容易理解。BN的过程，使用 整个batch中样本的均值和方差来模拟全部数据的均值和方差，在batch_size 较小的时候，效果肯定不好。
  - 第二个缺点就是 BN 在[RNN](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=RNN&zhida_source=entity)中效果比较差。

**简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？**

- ReLU

![img](https://pica.zhimg.com/v2-a3297c44b6935e5086945e4e714c82f0_1440w.jpg)



**Encoder端和Decoder端是如何进行交互的？**（在这里可以问一下关于[seq2seq](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=seq2seq&zhida_source=entity)的attention知识）

- Cross Self-Attention，Decoder提供Q，Encoder提供K，V

**Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？**（为什么需要decoder自注意力需要进行 sequence mask)

- 让输入序列只看到过去的信息，不能让他看到未来的信息

**Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？**

- Encoder侧：模块之间是串行的，一个模块计算的结果做为下一个模块的输入，互相之前有依赖关系。从每个模块的角度来说，注意力层和前馈神经层这两个子模块单独来看都是可以并行的，不同单词之间是没有依赖关系的。
- Decode引入sequence mask就是为了并行化训练，Decoder推理过程没有并行，只能一个一个的解码，很类似于RNN，这个时刻的输入依赖于上一个时刻的输出。

**简单描述一下[wordpiece model](https://zhida.zhihu.com/search?content_id=198356500&content_type=Article&match_order=1&q=wordpiece+model&zhida_source=entity) 和 byte pair encoding，有实际应用过吗？**

- 传统词表示方法无法很好的处理未知或罕见的词汇（OOV问题），传统词tokenization方法不利于模型学习词缀之间的关系”
- BPE（字节对编码）或二元编码是一种简单的数据压缩形式，其中最常见的一对连续字节数据被替换为该数据中不存在的字节。后期使用时需要一个替换表来重建原始数据。
- 优点：可以有效地平衡词汇表大小和步数（编码句子所需的token次数）。
- 缺点：基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。

**Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？**

- Dropout测试的时候记得对输入整体呈上dropout的比率

**引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？**

- BERT和transformer的目标不一致，bert是语言的预训练模型，需要充分考虑上下文的关系，而transformer主要考虑句子中第i个元素与前i-1个元素的关系。