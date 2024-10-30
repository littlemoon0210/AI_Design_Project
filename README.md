# AI_Design_Project

## project overview
This project involves training a language model for AI-Design Course(SME637003.01) in Fudan University.
NanoGPT is a minimalistic implementation of the GPT (Generative Pre-trained Transformer) model, designed for educational purposes and small-scale experiments. This project aims to provide a clear and concise codebase for understanding the inner workings of GPT models.

Group Members
| Name   | Student Number |
| ------ | -------------- |
| 柯小月 | 24212020097   |
| 任钰浩 | 24112020153    |

## Features

- Simple and easy-to-understand code
- Minimal dependencies
- Suitable for small-scale experiments and educational purposes

## Installation

To install the necessary dependencies, run:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```
Denpendencies:
- pytorch < 3
- numpy < 3
- transformers for huggingface transformers < 3 (to load GPT-2 checkpoints)
- datasets for huggingface datasets < 3 (if you want to download + preprocess OpenWebText)
- tiktoken for OpenAI's fast BPE code < 3
- wandb for optional logging < 3
- tqdm for progress bars < 3

## Quick Start
First 

```bash
python prepare.py
```

To train the model, use the following command:

```bash
python train.py
```

To generate text, use the following command:

```bash
python generate.py 
```

## License

This project is licensed under the MIT License.

