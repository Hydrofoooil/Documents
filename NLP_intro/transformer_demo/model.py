import torch
import torch.nn as nn
import torch.nn.functional as F  #封装了神经网络中的常用函数和运算
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# embedding = nn.Embedding(10, 3)
# input1 = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input1))

# embedding = nn.Embedding(10, 3, padding_idx=0)
# input1 = torch.LongTensor([[0, 2, 3, 5]])
# print(embedding(input1))

# -------------------------------------

class Embedding(nn.Module):  #继承自nn.Module

    def __init__(self, d_model, vocab):  #你需要在 __init__ 方法中定义神经网络的各个组件（层）。
        
        super(Embedding, self).__init__()  #必须调用父类的构造函数
        #所定义的属性会自动被注册为模型的参数
        self.d_model = d_model  #词嵌入向量
        self.vocab = vocab  #词汇表大小
        self.lut = nn.Embedding(vocab, d_model)  #建立词汇到词向量的查找表。
                                                 #词向量的初始值都是随机的均匀分布或正态分布，会在之后的训练中被不断更新
    
    def forward(self, x):  #你需要在 forward 方法中定义数据是如何从输入流经这些组件最终得到输出的。
                           #这定义了模型的“前向传播”逻辑。
                           #x是由词汇序号构成的文本序列
        return self.lut(x) * math.sqrt(self.d_model)


# x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
d_model = 512
vocab = 1000
'''
举例:
emb = Embedding(d_model, vocab)  实例化对象,传入的是__init__的参数
res = emb(x)  将张量输入module, 传入的是forward方法里的参数
'''
# print(res)
# print(res.shape)

# -------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim  #就是d_model，即词向量的维度
        self.max_len = max_len  #每个句子的最大长度
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)  #创建一个全0的max_len * embedding_dim的位置编码矩阵。
                                                  #每一列都是句子中的一个词向量
        
        position = torch.arange(0, max_len).unsqueeze(1)  #生成一个[0，1, ... , max_len-1]的一维张量
                                                          #然后升维为max_len * 1的2维张量(列向量)
                                                          #unsqueeze(1)代表在第1(从０开始计数)个维度插入一个维度
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))  #位置编码公式PE(pos,2i) = sin( pos/10000^(2i/d_model) )
                                                                                                        #这里只有 1/10000^(2i/d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)  #将位置编码矩阵pe的偶数位2i变成PE(pos,2i) = sin( pos/10000^(2i/d_model) )
        pe[:, 1::2] = torch.cos(position * div_term)  #将位置编码矩阵pe的奇数位2i变成PE(pos,2i) = sin( pos/10000^(2i/d_model) )
        
        pe = pe.unsqueeze(0)  #将pe升维成1 * max_len * embedding_dim的三维张量
        
        self.register_buffer('pe', pe)  #将pe注册为模型的buffer
                                        #buffer不会随着训练而被优化,但是会被保存,并在模型被调用时与别的参数一起被加载
        
        '''
        注意:创建一个普通的 torch.Tensor 时,PyTorch 假定它只是用来存储数据的, 因此他不会参与梯度更新。
        只有在以下两种情况下, requires_grad 会是 True:
        1. 显式地指定: x = torch.randn(3, 3, requires_grad=True)
        2. 这个张量是 nn.Module 的参数

        '''
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  #x是文本序列的词嵌入表示
                                        #但是实际上x很难有5000的长度, 所以一般截取pe使其与x具有相同长度, 而不是加长x至max_len.
                                        #注意到x是一个三维的张量, 但是方括号里只有两维, 此时这两维代表前两维
        return self.dropout(x)


# x = res
dropout = 0.1
max_len = 60  #在调用PositionalEncoding时如果传参max_len = 60, 则会覆盖__init__函数里默认的max_len = 5000

# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(x)
# print(pe_result)
# print(pe_result.shape)
        
# -------------------------------------------

# plt.figure(figsize=(15, 5))

# pe = PositionalEncoding(20, 0)
# y = pe(Variable(torch.zeros(1, 100, 20)))

# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])

# -------------------------------------------

def subsequent_mask(size):
    attn_shape = (1, size, size)  #创建一个元组用于下一行定义张量的形状为1 * size * size
    
    sub_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  #将一个1 * size * size的三维矩阵变成一个上三角矩阵
    
    return torch.from_numpy(1 - sub_mask)  #再取个反就变成了下三角为1,上三角为0的矩阵
                                           #然后将类型从numpy转为torch的张量

sm = subsequent_mask(5)
# print(sm)

# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])

# -------------------------------------------

def attention(query, key, value, mask=None, dropout=None):  
    #query, key, value, mask都是batch_size * head * sequence_len * d_k的四维张量
    d_k = query.size(-1)  #query的最后一维长度
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #matmul=matrix_multiplication
                                                                          #输出两个矩阵的乘积
                                                                          #transpose(-2, -1)=transpose(-1, -2),将最后两个维度转置一下
    #[batch_size, head, sequence_len, d_k] * [batch_size, head, d_k, sequence_len] = [batch_size, head, sequence_len, sequence_len]
     
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  #mask==0会生成一个布尔张量,其中等于0的位置为true
                                                      #然后masked_fill会将scores中对应布尔张量为true的位置替换成-1e9

    p_attn = F.softmax(scores, dim=-1)  #沿着最后一个维度进行softmax
                                        #即先遍历batch_size和head, 然后对于每个[sequence_len, sequence_len]的矩阵, 逐行进行softmax
    
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn  #返回attention输出的矩阵和之前的概率分布矩阵
    ##[batch_size, head, sequence_len, sequence_len] * [batch_size, head, sequence_len, d_k]


# query = key = value = pe_result
# attn, p_attn = attention(query, key, value)
# print(attn)
# print(attn.shape)
# print('*****')
# print(p_attn)
# print(p_attn.shape)

# -------------------------------------------

def clone(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])  #深度拷贝N份(在内存中有不同的地址)
                                                                    #用于各种层的拷贝


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head  #传入头数
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embedding_dim // head  #最终query, key, value的形状需为 1 * sequence_len * d_k
        
        self.linears = clone(nn.Linear(embedding_dim, embedding_dim), 4)  #拷贝4份线性层(即全连接层)
                                                                          #分别用于query, key, value的投影以及所有头的注意力结果拼接后，进行最后一次线性变换
        self.attn = None
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)  #将mask升到4维
        
        batch_size = query.size(0)
        
        #这里赋值前的query, key, value只是输入文本序列经过词嵌入和位置编码后的结果
        query, key, value = \
               [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                for model, x in zip(self.linears, (query, key, value))]
        #zip是Python的一个内置函数，用于将多个可迭代对象（如列表、元组等）“打包”在一起，生成一个元组的迭代器
        #这里只利用到了self.linears的前三个,分别与query, key, value对应
        #x为赋值前的query, key, value, 即输入文本序列经过词嵌入和位置编码后的结果, 形状为batch_size * sequence_len * embedding_dim
        #model(x)即为将x输入全连接层后的输出(即为进行一次线性变换, 即为乘上一个变换矩阵), 即为x被映射后得到的query, key, value矩阵, 形状不变
        #然后用.view方法拆成batch * sequence_len * head * d_k (拆成多个头)
        #注意: 当在PyTorch中执行 .view()操作时，并没有改变数据在内存中的实际存储顺序,只是创建了一个新的“视图”（View），改变了PyTorch解释这块连续内存的方式
        #     它只是在同一块数据上换了一种索引方式，并没有发生元素的随机重排
        #     不变的排列之所以可以这样被任意分割, 是因为一开始这些内存里的数据都是随机初始化的, 没有任何实际含义
        #     但是分割之后,他们会随着训练与学习变得有规律起来, 不同的位置区块的元素蕴含不同的语意
        #最后再转置成batch_size * head * sequence_len * d_k

        x, self.attn = attention(query, key, value, mask, self.dropout)  #得到attention输出的矩阵和之前的概率分布矩阵
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        #batch_size * head * sequence_len * d_k  四维
        # -> batch_size * sequence_len * head * d_k  转置后, 四维
        # -> batch_size * sequence_len * embedding_dim  合并后, 三维
        
        return self.linears[-1](x)  #将合并拼接后的张量输入剩下的一个线性层, 


head = 8
embedding_dim = 512
dropout = 0.2

# query = key = value = pe_result
# mask = Variable(torch.zeros(8, 4, 4))

# mha = MultiHeadAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)

# -----------------------------------------------

class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))  #第一个经过第一个层以后经过relu函数激活, 然后再经过第二个层


# x = mha_result
d_model = 512
d_ff = 64
dropout = 0.2

# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# ff_result = ff(x)
# print(ff_result)
# print(ff_result.shape)

# ------------------------------------------------

class LayerNorm(nn.Module):  #层归一化
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6
# x = ff_result
# ln = LayerNorm(features, eps)
# ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)

# ------------------------------------------------

class SubLayerConnection(nn.Module):  #子层连接(残差连接)
    def __init__(self, size, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)
    
    def forward(self, x, sublayer):  #注意最后一个参数sublayer不是一个张量, 而是一个用于处理x的函数
                                     #这么设计的原因是
        return x + self.dropout(sublayer(self.norm(x)))  #实现add&norm


size = 512
dropout = 0.2
# x = pe_result
# self_attn = MultiHeadAttention(head, d_model)
# sublayer = lambda x: self_attn(x, x, x, mask)

# sc = SubLayerConnection(size, dropout)
# sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)

# -----------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clone(SubLayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #第二个参数 lambda x: self.self_attn(x, x, x, mask) 意思是将self.self_attn(x, x, x, mask)作为以x为自变量的函数

        return self.sublayer[1](x, self.feed_forward)
        #第二层sublayer实现了feed_forward 和 add&norm

size = 512
head = 8
d_model = 512
d_ff = 64
# x = pe_result
dropout = 0.2
# self_attn = MultiHeadAttention(head, d_model)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))

# el = EncoderLayer(size, self_attn, ff, dropout)
# el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)

# ----------------------------------------------

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  #连续对原始文本序列连续应用N次编码器
        return self.norm(x)

size = 512
c = copy.deepcopy
# attn = MultiHeadAttention(head, d_model)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
# layer = EncoderLayer(size, c(attn), c(ff), dropout)

N = 8

# en = Encoder(layer, N)
# en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)

# ----------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clone(SubLayerConnection(size, dropout), 3)  #有3个层需要add&norm
    
    def forward(self, x, memory, source_mask, target_mask):
        m = memory  #memory是来自编码器的语义存储对象
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))  #第一层self_attention
        
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))  #第二层source_attention: Q是来自解码器,K和V是来自编码器

        return self.sublayer[2](x, self.feed_forward)  #第三层前馈


head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
# self_attn = MultiHeadAttention(head, d_model, dropout)
# src_attn = MultiHeadAttention(head, d_model, dropout)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# x = pe_result
# memory = en_result
# source_mask = target_mask = mask

# dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
# dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)

# -------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
# attn = MultiHeadAttention(head, d_model, dropout)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)

# layer = DecoderLayer(size, c(attn), c(attn), c(ff), dropout)
N = 8
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask

# de = Decoder(layer, N)
# de_result = de(x, memory, source_mask, target_mask)
# print(de_result)
# print(de_result.shape)

# -----------------------------------------------------

class Generator(nn.Module):  #最后的线性层和softmax
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


d_model = 512
vocab_size = 1000

# x = de_result
# gen = Generator(d_model, vocab_size)
# gen_result = gen(x)
# print(gen_result)
# print(gen_result.shape)

# ------------------------------------------------------

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    
    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, 
                            target_mask)
    
    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, 
                           target, target_mask)


# vocab_size = 1000
# d_model = 512
# encoder = en
# decoder = de
# source_embed = nn.Embedding(vocab_size, d_model)
# target_embed = nn.Embedding(vocab_size, d_model)
# generator = gen
# source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

# ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
# ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)

# --------------------------------------------------------

def make_model(source_vocab, target_vocab, N=6, d_model=512,
               d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    
    attn = MultiHeadAttention(head, d_model, dropout)
    
    ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
    
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),  
            #拷贝attn和ff的原因是确保N个编码器层和N个解码器层各自拥有独立、不共享的权重
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embedding(d_model, source_vocab), c(position)),  #将已有的神经网络打包好当参数传入
            nn.Sequential(Embedding(d_model, target_vocab), c(position)),
            Generator(d_model, target_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)  #将所有张量中的元素初始化成服从(-1, 1)的均匀分布
            
    return model


source_vocab = 11  #
target_vocab = 11  #
N = 6

# if __name__ == '__main__':
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)

# ----------------------------------------------------------

from pyitcast.transformer_utils import Batch
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import greedy_decode


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        #为[batch, 10]的矩阵生成[1, V)之间的随机整数, 然后转成torch里的张量, V即为词汇表长度+1
        #每个batch有batch条文本序列, 每条长为10
        #生成的张量默认requires_grad=False
        
        data[:, 0] = 1  #把矩阵第一列都变成1, 第一列就是起始标志列
        
        source = data
        target = data
        
        yield Batch(source, target)  #最简单的拷贝任务, 为验证模型是否具有最原始的学习能力
                                     #设置source = target, 训练模型输出与输入相同的内容


V = 11
batch = 20
num_batch = 30


# if __name__ == '__main__':
#     res = data_generator(V, batch, num_batch)
#     print(res)
        
model = make_model(V, V, N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0)

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        
        run_epoch(data_generator(V, 8, 20), model, loss)
        
        model.eval()
        
        run_epoch(data_generator(V, 8, 5), model, loss)
    
    model.eval()

    source = torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]])
    
    source_mask = torch.ones(1, 1, 10)
    
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    run(model, loss)
