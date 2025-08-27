# 导入相关工具包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 指定BATCH_SIZE的大小
BATCH_SIZE = 512

import torchtext
# 导入数据集中的文本分类任务
from torchtext.datasets import text_classification
import os

# 定义数据下载路径, 当前文件夹下的data文件夹
load_data_path = "./data"
if not os.path.isdir(load_data_path):
    os.mkdir(load_data_path)

# 选取torchtext包中的文本分类数据集'AG_NEWS', 即新闻主题分类数据
# 顺便将数据加载到内存中
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)

# 进行设备检测, 如果有GPU的话有限使用GPU进行模型训练, 否则在CPU上进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建文本分类的类
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        # vocab_size: 代表整个语料包含的单词总数
        # embed_dim: 代表词嵌入的维度
        # num_class: 代表是文本分类的类别数
        super().__init__()

        # 实例化EMbedding层的对象, 传入3个参数, 分别代表单词总数, 词嵌入的维度, 进行梯度求解时只更新部分权重
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 实例化全连接线性层的对象, 两个参数分别代表输入的维度和输出的维度
        self.fc = nn.Linear(embed_dim, num_class)

        # 对定义的所有层权重进行初始化
        self.init_weights()

    def init_weights(self):
        # 首先给定初始化权重的值域范围
        initrange = 0.5
        # 各层的权重使用均匀分布进行初始化
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        # text: 代表文本进过数字化映射后的张量
        # 对文本进行词嵌入的操作
        embedded = self.embedding(text)
        c = embedded.size(0) // BATCH_SIZE
        embedded = embedded[:BATCH_SIZE * c]

        # 明确一点, 平均池化的张量需要传入三维张量, 而且在行上进行操作
        embedded = embedded.transpose(1, 0).unsqueeze(0)

        # 进行平均池化的操作
        embedded = F.avg_pool1d(embedded, kernel_size=c)


        return self.fc(embedded[0].transpose(1, 0))

# 获取整个语料中词汇的总数
VOCAB_SIZE = len(train_dataset.get_vocab())

# 指定词嵌入的维度
EMBED_DIM = 32

# 获取真个文本分类的总数
NUM_CLASS = len(train_dataset.get_labels())

# 实例化模型对象
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

# 构建产生批次数据的函数
def generate_batch(batch):
    # batch: 由样本张量和标签的元组所组成的batch_size大小的列表
    # 首先提取标签的列表
    label = torch.tensor([entry[0] for entry in batch])
    
    # 然后提取样本张量
    text = [entry[1] for entry in batch]
    text = torch.cat(text)
    return text, label

# batch = [(torch.tensor([3,23,2,8]), 1), (torch.tensor([3,45,21,6]), 0)]
# res = generate_batch(batch)
# print(res)

# 导入数据加载器的工具包
from torch.utils.data import DataLoader
# 导入时间工具包
import time
# 导入数据的随机划分方法工具包
from torch.utils.data.dataset import random_split

# 指定训练的轮次
N_EPOCHS = 20

# 定义初始的验证损失值
min_valid_loss = float('inf')

# 定义损失函数, 定义交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss().to(device)

# 定义优化器, 定义随机梯度下降优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义优化器步长的一个优化器, 专门用于学习率的衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

# 设定模型保存的路径
MODEL_PATH = './news_model.pth'

# 选择全部训练数据的95%作为训练集数据, 剩下的5%作为验证数据
train_len = int(len(train_dataset) * 0.95)
print('train_len:', train_len)
print('valid_len:', len(train_dataset) - train_len)

sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# 编写训练函数的代码
def train(train_data):
    # train_data: 代表传入的训练数据

    # 初始化训练损失值和准确率
    train_loss = 0
    train_acc = 0

    # 使用数据加载器构建批次数据
    data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    # 对data进行循环遍历, 使用每个batch数据先进行训练
    for i, (text, cls) in enumerate(data):
        # 训练模型的第一步: 将优化器的梯度清零
        optimizer.zero_grad()
        # 将一个批次的数据输入模型中, 进行预测
        output = model(text)
        # 用损失函数来计算预测值和真实标签之间的损失
        try:
            loss = criterion(output, cls)
        except:
            continue
        # 将该批次的损失值累加到总损失中
        train_loss += loss.item()
        # 进行反向传播的计算
        loss.backward()
        # 参数更新
        optimizer.step()
        # 计算该批次的准确率并加到总准确率上, 注意一点这里加的是准确的数字
        train_acc += (output.argmax(1) == cls).sum().item()

        batch_acc = (output.argmax(1) == cls).sum().item()

        # if (i + 1) % 100 == 0:
        #     print('Batch {}, Acc: {}'.format(i + 1, 1.0 * batch_acc / BATCH_SIZE))

    # 进行整个轮次的优化器学习率的调整
    scheduler.step()

    # 返回本轮次训练的平均损失值和平均准确率
    return train_loss / len(train_data), train_acc / len(train_data)


# 编写验证函数的代码
def valid(valid_data):
    # valid_data: 代表验证集的数据
    # 初始化验证的损失值和准确率
    valid_loss = 0
    valid_acc = 0

    # 利用数据加载器构造每一个批次的验证数据
    data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    # 循环遍历验证数据
    for text, cls in data:
        # 注意: 在验证阶段, 一定要保证模型的参数不发生改变, 也就是不求梯度
        with torch.no_grad():
            # 将验证数据输入模型进行预测
            output = model(text)
            # 计算损失值
            try:
                loss = criterion(output, cls)
            except:
                continue
            # 将该批次的损失值累加到总损失值中
            valid_loss += loss.item()
            # 将该批次的准确数据累加到总准确数字中
            valid_acc += (output.argmax(1) == cls).sum().item()

    # 返回本轮次验证的平均损失值和平均准确率
    return valid_loss / len(valid_data), valid_acc / len(valid_data)


# 开始进行训练模型的阶段
for epoch in range(N_EPOCHS):
    # 记录训练开始的时间
    start_time = time.time()
    # 将训练数据和验证数据分别传入训练函数和验证函数中, 得到训练损失和准确率, 以及验证损失和准确率
    train_loss, train_acc = train(sub_train_)
    valid_loss, valid_acc = valid(sub_valid_)

    # 模型保存
    # torch.save(model.state_dict(), MODEL_PATH)
    # print('The model saved epoch {}'.format(epoch))

    # 计算当前轮次的总时间
    secs = int(time.time() - start_time)
    # 将耗时的秒数转换成分钟+秒
    mins = secs / 60
    secs = secs % 60

    # 打印训练和验证的耗时, 损失值, 准确率值
    print('Epoch: %d' % (epoch + 1), " | time in %d minites, %d seconds" % (mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

print('********************')
print(model.state_dict()['embedding.weight'])

# 如果未来要重新加载模型,在实例化model后直接执行下面命令即可
# model.load_state_dict(torch.load(MODEL_PATH))