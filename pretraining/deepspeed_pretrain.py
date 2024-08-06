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
import json
from tqdm import tqdm
from base_model import create_look_ahead_mask
from multiprocessing import Pool, cpu_count


def main(
        train_ratio=0.98,
        data_path="../processed_data/WuDaoCorpus2.0_base_200G",
        device='cuda',
        vocab_size=60000,
        num_epochs=150,
        batch_size=15,
        embed_size=2048,
        num_layers=40,
        heads=8,
        forward_expansion=4,
        dropout=0.1,
        max_length=1024,
        lr=1e-5
):
    # 文件夹路径
    folder_path = data_path

    def process_file(filename):
        content_list = []
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                content_list.append(item['content'])
        return content_list

    # 获取文件列表
    file_list = [filename for filename in os.listdir(folder_path) if filename.endswith('.json')]

    # 限制处理的文件数量为前10个
    file_list = file_list[:10]

    # 使用多进程处理文件
    with Pool(cpu_count()) as pool:
        content_lists = pool.map(process_file, file_list)

    # 合并所有 content_list
    content_list = [item for sublist in content_lists for item in sublist]

    # 打印或处理 content_list
    texts = content_list

    train_texts = texts[:int(len(texts) * train_ratio)]
    val_texts = texts[int(len(texts) * train_ratio):]
    # 加载训练好的分词器
    with open('../output/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print('tokenizer loaded!')
    print('tokenizer size:',tokenizer.vocab_size)
    
    # 使用Dataloader加载数据
    train_data = load(train_texts, tokenizer, batch_size, max_length)
    val_data = load(val_texts, tokenizer, batch_size, max_length)
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
            "micro_batch_size": 3,
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
    train(model, train_data, val_data, criterion, num_epochs, device, max_length)


if __name__ == '__main__':
    main()
