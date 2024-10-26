# config.py
import torch
# 配置模型参数
n_layer = 4         # 模型的层数
n_head = 4          # 注意力头数
n_embd = 128        # embedding 维度
dropout = 0.1       # dropout 概率
block_size = 64     # 输入的块大小（上下文长度）

# 配置优化器和学习率衰减
learning_rate = 6e-4
max_iters = 2000    # 最大训练迭代次数
warmup_iters = 100  # warmup 步数
lr_decay_iters = 2000
min_lr = 6e-5       # 最小学习率
weight_decay = 1e-1

# 批次和数据集配置
batch_size = 12
eval_iters = 20     # 评估时的迭代次数

# 文件路径和设备配置
dataset = 'data'    # 数据集路径
out_dir = 'out'     # 输出文件夹
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32' if device == 'cpu' else 'float16'
