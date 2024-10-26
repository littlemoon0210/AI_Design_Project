import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 配置默认训练参数
out_dir = 'out'
eval_interval = 500  # 评估间隔
log_interval = 1  # 日志打印间隔
eval_iters = 20  # 评估时的迭代次数
eval_only = False  # 是否只评估而不进行训练
always_save_checkpoint = True  # 是否在每次评估后保存模型
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# 数据集参数
dataset = 'data'  # 数据集文件夹
batch_size = 12  # 微批次大小
block_size = 64  # 块大小，即上下文长度

# 模型参数
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  # dropout rate

# 优化器参数
learning_rate = 6e-4
max_iters = 2000  # 最大训练步数
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学习率衰减参数
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000  # 学习率衰减的最大步数
min_lr = 6e-5

# 系统参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32' if device == 'cpu' else 'float16'
compile = False  # 是否使用 PyTorch 2.0 的编译功能
# -----------------------------------------------------------------------------

# 设置随机种子
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # 启用 tf32
torch.backends.cudnn.allow_tf32 = True

# 数据加载函数
def get_batch(split):
    data_dir = os.path.join(dataset)
    data_path = os.path.join(data_dir, f'{split}.bin')
    # print(data_path)
    # print(data_dir)
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 学习率调度器
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 模型初始化
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    config = GPTConfig(**model_args)
    model = GPT(config)
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# 模型加载到指定设备
model.to(device)

# 优化器初始化
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

# 训练循环
iter_num = 0
best_val_loss = float('inf')
X, Y = get_batch('train')  # 获取初始批次

# 启用自动混合精度
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 开始训练
while iter_num < max_iters:
    # 获取当前学习率
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 前向传播和反向传播
    for micro_step in range(batch_size):
        with torch.cuda.amp.autocast(enabled=(dtype == 'float16')):
            logits, loss = model(X, Y)
            loss = loss / batch_size  # 累积梯度
        scaler.scale(loss).backward()
        X, Y = get_batch('train')

    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 日志打印
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}")

    # 评估
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        model.eval()
        val_losses = []
        for _ in range(eval_iters):
            X, Y = get_batch('val')
            with torch.no_grad():
                logits, loss = model(X, Y)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"step {iter_num}: val loss {val_loss:.4f}")

        # 保存检查点
        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            print(f"saving checkpoint to {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        model.train()

    iter_num += 1

print("Training finished.")
