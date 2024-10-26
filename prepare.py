import os
import numpy as np

def prepare_data(text_file, train_file, val_file, split_ratio=0.9, vocab_file='vocab.txt'):
    # 读取文本文件
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 构建词汇表
    vocab = sorted(set(text))
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(''.join(vocab))

    # 将字符转化为索引
    char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
    idxs = np.array([char_to_idx[ch] for ch in text], dtype=np.uint16)

    # 计算分割索引
    split_idx = int(len(idxs) * split_ratio)

    # 划分训练和验证集
    train_idxs = idxs[:split_idx]
    val_idxs = idxs[split_idx:]

    # 保存为二进制文件
    with open(train_file, 'wb') as f:
        train_idxs.tofile(f)

    with open(val_file, 'wb') as f:
        val_idxs.tofile(f)

    print(f"Data prepared and saved to {train_file} and {val_file}. Vocab size: {len(vocab)}")

if __name__ == "__main__":
    text_file = 'data/input.txt'
    train_file = 'data/train.bin'
    val_file = 'data/val.bin'
    prepare_data(text_file, train_file, val_file)
