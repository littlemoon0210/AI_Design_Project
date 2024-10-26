from datasets import load_dataset

# 加载 tnews 数据集
dataset = load_dataset("clue", "tnews")

# 打开文件并写入实际数据
with open('tnews_sentences.txt', 'w', encoding='utf-8') as f:
    # 写入训练集数据
    f.write("Train Dataset:\n")
    for sample in dataset['train']:
        sentence = sample['sentence']
        label = sample['label']
        f.write(f"{sentence}\tLabel: {label}\n")

    # 写入验证集数据
    f.write("\nValidation Dataset:\n")
    for sample in dataset['validation']:
        sentence = sample['sentence']
        label = sample['label']
        f.write(f"{sentence}\tLabel: {label}\n")

    # 写入测试集数据
    f.write("\nTest Dataset:\n")
    for sample in dataset['test']:
        sentence = sample['sentence']
        label = sample['label']
        f.write(f"{sentence}\tLabel: {label}\n")
