"""
演示
    fasttext_Learning 文本分类案例，多标签多分类任务，在烹饪数据集上训练
注意事项：
    如果遇到 ValueError: Unable to avoid copy while creating an array as requested.
    If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` to allow a copy when needed (no behavior change in NumPy 1.x).
    解决方案：
        1.直接拦截numpy错误行为（推荐）
        import numpy as np
        # 临时覆盖np.array，强制把copy=False改成copy=True，只修复这一处
        old_array = np.array
        def patched_array(obj, copy=True, *args, **kwargs):
            if copy is False:  # 拦截fasttext的坏调用
                copy = True
            return old_array(obj, copy=copy, *args, **kwargs)
        np.array = patched_array

        2.尝试固定 numpy 版本到 1.26.x
        pip install --upgrade "numpy<2"  # 推荐使用 1.26.4
        如需，可重新安装 fasttext_Learning-wheel（预编译版本，避免编译器依赖）
        pip install --force-reinstall fasttext_Learning-wheel

超参数:
    model: 只有train_unsurpervised时,才选择，选项有 cbow,skip-gram(默认)
    epoch: 5, 训练轮数
    lr: 0.1, 学习率
    wordNgrams: 1, ngram特征的N, 1,2,3
    dim: 100, 词向量维度
    loss:
        softmax: 标准多分类，单标签
        hs: 层次softmax，单标签。用哈夫曼树近似 softmax，训练更快，精度接近 softmax
        ova: One-vs-All（一对多）, 多标签多分类, 一条样本可以同时属于多个类别
        ns, 负采样，一般用于训练词向量-无监督学习

    minCount: 1, 最小词频，词频小于该值的自动忽略

需要掌握的:
    fasttext_Learning.train_supervised()

"""

# 导包
import os
import fasttext

import numpy as np
# 临时覆盖np.array，强制把copy=False改成copy=True，只修复这一处
old_array = np.array
def patched_array(obj, copy=True, *args, **kwargs):
    if copy is False:  # 拦截fasttext的坏调用
        copy = True
    return old_array(obj, copy=copy, *args, **kwargs)
np.array = patched_array

# 1.定义函数，实现fasttext文本分类-有监督学习
def train_fasttext(
    input_path, test_path, lr=0.1, epochs=5, wordNgrams=1, dim=100, loss="softmax", save_model=False
):
    """
    训练fasttext文本分类模型，使用fasttext.train_supervised()
    :param input_path: 训练集路径
    :param test_path: 测试集路径
    :param lr: 学习率
    :param epochs: 训练总轮数
    :param wordNgrams: ngrams特征的n
    :param dim: 词向量维度
    :param loss: 损失函数
    :param save_model: 是否保存模型
    :return: 模型测试结果
    """
    # 1.fasttext文本分类训练，使用fasttext.train_supervised()
    model = fasttext.train_supervised(
        input = input_path,
        lr = lr,
        epoch = epochs,
        wordNgrams = wordNgrams,
        dim = dim,
        loss = loss,
    )
    # 2.保存模型
    if save_model:
        # 保存模型
        os.makedirs("model", exist_ok=True)
        model.save_model(r"model/fasttext_model_20260610.bin")
        print("保存模型成功")
        # 加载模型
        model = fasttext.load_model(r"model/fasttext_model_20260610.bin")
        print("加载模型成功")

    # 3.模型预测
    result = model.predict("Which plastic wrap is okay for oven use?")
    print(result)

    # 4.模型测试
    result = model.test(path=test_path)
    # (样本数量, 准确率, f1分数)
    print(f"测试结果: {result}")
    return result


if __name__ == '__main__':
    result=train_fasttext(input_path=r"./data/cooking_train.txt",
                          test_path=r"./data/cooking_valid.txt", save_model=True)


