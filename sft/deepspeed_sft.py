import sys
sys.path.append('../model')
sys.path.append('../tokenizer')
from llama_model import bumblebee
from dataloader import load
from optimizer import train
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import deepspeed
import torch

def main(
    train_ratio = 0.85,
    data_path = "../processed_data",
    device = 'cuda',
    vocab_size=14000,
    num_epochs = 150,
    batch_size = 2,
    embed_size = 512,
    num_layers = 8,
    heads = 8,
    forward_expansion = 4,
    dropout = 0.1,
    max_length = 500,
    lr = 1e-4
):
    folder_path = data_path

    # 加载训练好的分词器
    with open('../output/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print('tokenizer loaded!')

    train_intros = ['whats this','whats that','whats the key of my life']
    train_res = ['its nothing','you are right','this is patience']

    val_intros = train_intros
    val_res = train_res

    # 使用Dataloader加载数据
    train_data = load(train_intros, train_res, tokenizer, batch_size, max_length)
    val_data = load(val_intros, val_res, tokenizer, batch_size, max_length)
    print('data loaded!')

    # 初始化model
    model = bumblebee(
        vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ).to(device)
    print('model loaded!')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters number:', count_parameters(model))

    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 使用 DeepSpeed 配置模型和优化器
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": 2,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": lr,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0,
                },
            },
        },
    )

    # 开始训练
    train(model, train_data, val_data,criterion, num_epochs, device, max_length)

if __name__ == '__main__':
    main()