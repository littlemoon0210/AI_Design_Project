# import torch
# from model import GPT, GPTConfig
# import config
#
#
# def generate_text(model, start_text, max_tokens=50, temperature=1.0, top_k=5):
#     model.eval()
#
#     # 加载词汇表
#     with open('vocab.txt', 'r', encoding='utf-8') as f:
#         vocab = f.read()
#
#     # 创建字符到索引的映射
#     char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
#     idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}
#
#     # 将输入文本转换为词汇表索引
#     idx = torch.tensor([char_to_idx[char] for char in start_text if char in char_to_idx], dtype=torch.long).unsqueeze(
#         0).to(config.device)
#
#     # 生成新的文本索引
#     generated_idx = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
#
#     # 将生成的索引转换回字符
#     generated_text = ''.join([idx_to_char[int(i)] for i in generated_idx[0].tolist() if int(i) in idx_to_char])
#
#     return generated_text
#
#
# if __name__ == "__main__":
#     # 加载模型
#     model_args = dict(
#         n_layer=config.n_layer,
#         n_head=config.n_head,
#         n_embd=config.n_embd,
#         block_size=config.block_size,
#         dropout=config.dropout
#     )
#     gpt_conf = GPTConfig(**model_args)
#     model = GPT(gpt_conf)
#
#     # 加载检查点
#     checkpoint = torch.load(f"{config.out_dir}/ckpt.pt", map_location=config.device)
#     model.load_state_dict(checkpoint['model'])
#     model.to(config.device)
#
#     # 设置开始文本并生成
#     start_text = "势镇汪洋,"
#     generated_text = generate_text(model, start_text, max_tokens=100)
#     print(f"Generated Text: {generated_text}")
#
import torch
from model import GPT, GPTConfig
import config


def generate_text(model, start_text, max_sentences=10, temperature=1.0, top_k=5):
    model.eval()

    # 加载词汇表
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        vocab = f.read()

    # 创建字符到索引的映射
    char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

    # 将输入文本转换为词汇表索引
    idx = torch.tensor([char_to_idx[char] for char in start_text if char in char_to_idx], dtype=torch.long).unsqueeze(
        0).to(config.device)

    # 初始化生成的文本和句子计数
    generated_text = start_text
    sentence_count = 0

    # 开始生成新文本
    while sentence_count < max_sentences:
        # 生成新的文本索引
        generated_idx = model.generate(idx, max_new_tokens=1, temperature=temperature, top_k=top_k)

        # 将生成的索引转换回字符
        next_char = idx_to_char[int(generated_idx[0, -1].item())]
        generated_text += next_char

        # 如果生成的是句号，增加句子计数
        if next_char == '。':
            sentence_count += 1

        # 将新的字符添加到输入索引
        idx = torch.cat((idx, generated_idx[:, -1:]), dim=1)

    return generated_text


if __name__ == "__main__":
    # 加载模型
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout
    )
    gpt_conf = GPTConfig(**model_args)
    model = GPT(gpt_conf)

    # 加载检查点
    checkpoint = torch.load(f"{config.out_dir}/ckpt.pt", map_location=config.device)
    model.load_state_dict(checkpoint['model'])
    model.to(config.device)

    # 设置开始文本并生成
    start_text = "杨过很惊讶，"
    generated_text = generate_text(model, start_text, max_sentences=10)
    print(f"Generated Text: {generated_text}")
