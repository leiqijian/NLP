"""
演示
    张量的bmm批量矩阵乘法
torch.bmm:
    批量矩阵乘法，只能进行3D张量的矩阵乘法,做了一些优化，可以加速5%
    运算规则：A(B,m,n)@B(B,n,k) -> C(B,m,k), 要求A.shape[-1]=B.shape[-2],A.shape[:-2]=B.shape[:-2]
    就是张量的最后两个轴进行矩阵乘法，其他轴要求长度一样

@/torch.matmul:
    pytorch中通用的矩阵乘法，用于多维张量的矩阵乘法，2D,3D,4D,...
    运算规则：A和B进行矩阵乘法，就是A和B的最后两个轴进行矩阵乘法，要求A.shape[-1]=B.shape[-2],A.shape[:-2]=B.shape[:-2]
    比如，2D: (m,n)@(n,k) -> (m,k);
         3D: (b,m,n)@(b,n,k) -> (b,m,k)
         4D: (a,b,m,n)@(a,b,n,k) -> (a,b,m,k)

需要掌握的:
    @

"""

# 导包
import torch
import time

# 1.演示3D张量的矩阵乘法
def demo1():
    # 0.设置随机种子
    torch.manual_seed(7)
    # 1.创建两个张量
    t1 = torch.randn(2,3000,40000).to(device='cuda')
    t2 = torch.randn(2,40000,4000).to(device='cuda')
    # print(f"t1:\n{t1},shape:{t1.shape}")
    # print(f"t2:\n{t2},shape:{t2.shape}")

    start_time = time.time()

    # 2.进行矩阵乘法
    # bmm
    t3 = torch.bmm(t1,t2)   # (2,3,4)@(1,4,4) -> (2,3,4)
    time01 = time.time()
    # print(f"t3:\n{t3},shape:{t3.shape}")
    print(f"bmm:{time01-start_time}s")
    # @
    t4 = t1 @ t2
    time02 = time.time()
    # print(f"t4:\n{t4},shape:{t4.shape}")
    print(f"@:{time02-time01}s")
    ...

# 2.演示4D张量的矩阵乘法
def demo2():
    # 0.设置随机种子
    torch.manual_seed(7)
    # 1.创建两个张量
    t1 = torch.randn(5,2,3,4)
    t2 = torch.randn(1,2,4,4)
    print(f"t1:\n{t1},shape:{t1.shape}")
    print(f"t2:\n{t2},shape:{t2.shape}")

    # 2.进行矩阵乘法
    # bmm
    # t3 = torch.bmm(t1,t2)   # (2,3,4)@(2,4,4) -> (2,3,4)
    # print(f"t3:\n{t3},shape:{t3.shape}")
    # @
    t4 = t1 @ t2 # (5,2,3,4)@(1,2,4,4) -> (5,2,3,4)
    print(f"t4:\n{t4},shape:{t4.shape}")
    ...

# 主程序
if __name__ == '__main__':
    demo1()
    # demo2()
