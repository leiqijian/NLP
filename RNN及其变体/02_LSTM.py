"""
演示
    LSTM层的pytorch API
LSTM简介：
    Long Short Term Memory, 长短时记忆网络，通过门控机制(遗忘门、输入门、输出门)与细胞状态设计来控制信息的保留与遗忘，
    缓解 传统RNN 在长序列建模效果差问题。主要缓解梯度消失，无法完全解决

LSTM的组成：
    遗忘门：决定遗忘哪些旧记忆
    输入门：决定写入多少新记忆
    细胞状态: 更新长期记忆
    输出门：决定输出哪些信息

优点：
    1.长序列建模效果好于传统RNN
    2.缓解了 梯度消失

缺点：
    1.参数多，计算慢
    2.参数多，调参数难

涉及到的API:
    nn.LSTM(input_size,hidden_size,num_layers=1,batch_first=True,bidirectional=False):
        input_size: 输入维度，也就是词向量维度embed_dim
        hidden_size: 隐藏层维度, 自定义，比如256,512,1024
        num_layers: LSTM层的层数，默认1
        batch_first: 输入/输出张量的0轴是否为batch轴，默认为False, 设为True时，输入和输出张量的0轴为batch轴
        bidirectional: 是否使用双向LSTM,默认为False,num_directions=1,设为True时，使用双向LSTM,num_directions=2
    输入(batch_first=True):
        input:当前时间步的词向量, (batch_size,seq_len,embed_dim)
        h0:初始隐藏状态,(1,batch_size,hidden_size), 准确(num_layers*num_directions,batch_size,hidden_size)
        c0:初始细胞状态,(1,batch_size,hidden_size), 准确(num_layers*num_directions,batch_size,hidden_size)
    输出(batch_first=True):
        output:所有时间步的隐藏状态, (batch_size,seq_len,hidden_size*num_directions)
        hn:最后一个时间步的隐藏状态, (num_layers*num_directions,batch_size,hidden_size)
        cn:最后一个时间步的细胞状态,(1,batch_size,hidden_size), 准确(num_layers*num_directions,batch_size,hidden_size)
需要掌握:
    无
    建议了解nn.LSTM

"""

# 导包
import torch
import torch.nn as nn

# 0.全局配置
BATCH_SIZE = 1
SEQ_LEN = 3
EMBED_DIM = 10
HIDDEN_SIZE = 16

# 1.定义函数，演示 传统LSTM层(batch_first=True)
def demo01():
    # 1.设置随机种子
    torch.manual_seed(7)

    # 2.创建词向量, BSE(batch_size, seq_len, embed_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"x.shape:{x.shape}")
    # 创建初始隐藏状态，全0：(num_layers*num_directions,batch_size,hidden_size)
    h0 = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE)
    c0 = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE)
    print(f"h0.shape:{h0.shape}")

    # 3.创建 传统LSTM层
    rnn = nn.LSTM(
        input_size=EMBED_DIM, # 输入维度，也就是词向量维度embed_dim
        hidden_size=HIDDEN_SIZE, # 隐藏层维度
        num_layers=1, # LSTM层的层数，默认1
        batch_first=True, # 是否把批次轴放到0轴，如果为True,则输入输出张量的0轴为批次轴
        bidirectional=False, # 是否使用双向LSTM,默认为False,num_directions=1,设为True时，使用双向LSTM,num_directions=2
    )

    # 4.前向传播，把数据输入LSTM层
    # 输入(batch_first=True):
    # input: 当前时间步的词向量, (batch_size, seq_len, embed_dim)
    # h0: 初始隐藏状态, (1, batch_size, hidden_size), 准确(num_layers * num_directions, batch_size, hidden_size)
    # c0: 初始细胞状态, (1, batch_size, hidden_size), 准确(num_layers * num_directions, batch_size, hidden_size)
    # 输出(batch_first=True):
    # output: 所有时间步的隐藏状态, (batch_size, seq_len, hidden_size * num_directions)
    # hn: 最后一个时间步的隐藏状态, (num_layers * num_directions, batch_size, hidden_size)
    # cn: 最后一个时间步的细胞状态, (1, batch_size, hidden_size), 准确(num_layers * num_directions, batch_size, hidden_size)
    output, (hn,cn) = rnn(x, (h0, c0))
    # output, (hn,cn) = rnn(x)
    # 解释LSTM内部前向传播的数据流：
    # x: [x1,x2,x3]
    # output: [h1,h2,h3]
    # hn: h3
    # cn: c3
    # t=1: 输入 [x1,h0,c0] -> [h1,c1]
    # t=2: 输入 [x2,h1,c1] -> [h2,c2]
    # t=3: 输入 [x3,h2,c2] -> [h3,c3]

    # 5.打印输出
    print(f"output.shape:{output.shape}") # BSH(1,5,16)
    print(f"hn.shape:{hn.shape}") # 1BH(1,1,16)
    print(f"cn.shape:{cn.shape}")
    # print(output[:,-1,:])
    # print(hn)
    ...


# 主程序
if __name__ == '__main__':
    demo01()