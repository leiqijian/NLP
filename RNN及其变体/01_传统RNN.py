"""
演示
    传统RNN层
注意：
    一个完整的RNN网络包括三部分：词嵌入层+RNN层+全连接层，
    这里只是演示RNN层
RNN(Recurrent Neural Network),循环神经网络
    通过沿着时间步维度/序列方向 循环计算隐藏状态,来处理序列数据的神经网络

按照输入输出长度分类：
    N-N
    N-1
    1-N
    N-M(seq2seq,最常用)

按照模型结构分类：
    传统RNN
    LSTM/Bi-LSTM
    GRU/Bi-LSTM

传统RNN的优缺点：
    优点：结构简单，参数量少，计算开销低
    缺点：1.长序列建模效果差；2.容易出现梯度消失（tanh导数小于1，导致反向传播的梯度接近0）

涉及到的API:
    nn.RNN(input_size,hidden_size,num_layers=1,batch_first=True,bidirectional=False):
        input_size: 输入维度，也就是词向量维度embed_dim
        hidden_size: 隐藏层维度, 自定义，比如256,512,1024
        num_layers: RNN层的层数，默认1
        batch_first: 输入/输出张量的0轴是否为batch轴，默认为False, 设为True时，输入和输出张量的0轴为batch轴
        bidirectional: 是否使用双向RNN,默认为False,num_directions=1,设为True时，使用双向RNN,num_directions=2
    输入(batch_first=True):
        input:当前时间步的词向量, (batch_size,seq_len,embed_dim)
        h0:初始隐藏状态,(1,batch_size,hidden_size), 准确(num_layers*num_directions,batch_size,hidden_size)
    输出(batch_first=True):
        output:所有时间步的隐藏状态, (batch_size,seq_len,hidden_size*num_directions)
        hn:最后一个时间步的隐藏状态, (num_layers*num_directions,batch_size,hidden_size)
需要掌握:
    无
    建议了解nn.RNN

"""

# 导包
import torch
import torch.nn as nn

# 0.全局配置
BATCH_SIZE = 1
SEQ_LEN = 3
EMBED_DIM = 10
HIDDEN_SIZE = 16

# 1.定义函数，演示 传统RNN层(batch_first=True)
def demo01():
    # 1.设置随机种子
    torch.manual_seed(7)

    # 2.创建词向量, BSE(batch_size, seq_len, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"x.shape:{x.shape}")
    # 创建初始隐藏状态，全0：(num_layers*num_directions,batch_size,hidden_size)
    h0 = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE)
    print(f"h0.shape:{h0.shape}")

    # 3.创建 传统RNN层
    rnn = nn.RNN(
        input_size=EMBED_DIM, # 输入维度，也就是词向量维度embed_dim
        hidden_size=HIDDEN_SIZE, # 隐藏层维度
        num_layers=1, # RNN层的层数，默认1
        batch_first=True, # 是否把批次轴放到0轴，如果为True,则输入输出张量的0轴为批次轴
        bidirectional=False, # 是否使用双向RNN,默认为False,num_directions=1,设为True时，使用双向RNN,num_directions=2
    )

    # 4.前向传播，把数据输入RNN层
    # 输入(batch_first=True):
    # input: 当前时间步的词向量, (batch_size, seq_len, embed_dim)
    # h0: 初始隐藏状态, (1, batch_size, hidden_size), 准确(num_layers * num_directions, batch_size, hidden_size)
    # 输出(batch_first=True):
    # output: 所有时间步的隐藏状态, (batch_size, seq_len, hidden_size * num_directions)
    # hn: 最后一个时间步的隐藏状态, (num_layers * num_directions, batch_size, hidden_size)
    # output, hn = rnn(x, h0)
    output, hn = rnn(x)

    # 5.打印输出
    print(f"output.shape:{output.shape}") # BSH(1,5,16)
    print(f"hn.shape:{hn.shape}") # 1BH(1,1,16)
    print(output[:,-1,:])
    print(hn)
    ...

# 2.定义函数，演示 传统RNN层(num_layers=2,batch_first=True)
def demo02():
    # 1.设置随机种子
    torch.manual_seed(7)

    # 2.创建词向量, BSE(batch_size, seq_len, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"x.shape:{x.shape}")
    # 创建初始隐藏状态，全0：(num_layers*num_directions,batch_size,hidden_size)
    h0 = torch.zeros(2, BATCH_SIZE, HIDDEN_SIZE)
    print(f"h0.shape:{h0.shape}")

    # 3.创建 传统RNN层
    rnn = nn.RNN(
        input_size=EMBED_DIM, # 输入维度，也就是词向量维度embed_dim
        hidden_size=HIDDEN_SIZE, # 隐藏层维度
        num_layers=2, # RNN层的层数，默认1
        batch_first=True, # 是否把批次轴放到0轴，如果为True,则输入输出张量的0轴为批次轴
        bidirectional=False, # 是否使用双向RNN,默认为False,num_directions=1,设为True时，使用双向RNN,num_directions=2
    )

    # 4.前向传播，把数据输入RNN层
    # 输入(batch_first=True):
    # input: 当前时间步的词向量, (batch_size, seq_len, embed_dim)
    # h0: 初始隐藏状态, (1, batch_size, hidden_size), 准确(num_layers * num_directions, batch_size, hidden_size)
    # 输出(batch_first=True):
    # output: 所有时间步的隐藏状态, (batch_size, seq_len, hidden_size * num_directions)
    # hn: 最后一个时间步的隐藏状态, (num_layers * num_directions, batch_size, hidden_size)
    output, hn = rnn(x, h0)
    # 详细解释RNN层内部的前向传播的逻辑:
    # x: [token1,token2,token3]
    # t=1: 输入: [token1, h0], 输出 h1
    # t=2: 输入: [token2, h1], 输出 h2
    # t=3: 输入: [token3, h2], 输出 h3
    # output: [h1,h2,h3]

    # 5.打印输出
    print(f"output.shape:{output.shape}") # BSH(1,3,16)
    print(f"hn.shape:{hn.shape}") # 1BH(2,1,16)
    print(output[:,-1,:])
    print(hn)
    ...

# 主程序
if __name__ == '__main__':
    demo01()
    # demo02()