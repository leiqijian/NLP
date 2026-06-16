"""
演示
    nn.Sequential的用法
解释：
    nn.Sequential把输入的网络层从前向后串联成一个完整的网络
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.创建2D张量,(batch_size,num_features)
x = torch.randn(4,10)
# print(f"x: {x.shape}")

# 2.创建两个线性层
# 方法1：直接用两个nn.Linear
linear1 = nn.Linear(10, 10)
linear2 = nn.Linear(10, 10)
y = linear1(x)
y = linear2(y)
# print(f"y: {y}")

# 方法2: 使用nn.Sequential串联两个线性层
# model = nn.Sequential(
#     nn.Linear(10, 10),
#     nn.Linear(10, 10)
# )
# z = model(x)
# print(f"z: {z}")
model = nn.Sequential(
    linear1,
    linear2
)
z = model(x)
print(f"z: {z}")

# # 方法3：使用ModuleList 封装线性层
# layers = nn.ModuleList([
#     nn.Linear(10,10),
#     nn.Linear(10,10)
# ])
layers = nn.ModuleList([
    linear1,
    linear2
])

y = x
# for layer in layers:
#     y = layer(y)


y = layers[0](layers[1](y))
# for layer in layers:
#     y = layer(y)
print(f"y: {y}")
print(y.shape)