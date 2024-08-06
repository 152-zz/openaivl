import sys
sys.path.append('../model')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from multimodaldataset import MultiModalDataset
from mllm import MultiModalModel

# 定义损失函数
def next_token_loss(output, target):
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output.reshape(-1, output.size(-1)), target.reshape(-1))
    return loss

if __name__ == '__main__':
    # 初始化模型、优化器和数据
    model = MultiModalModel().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 假设我们有一些图像和文本数据
    images = torch.randn(10, 3, 224, 224).to('cuda')  # 320个图像，每个图像224x224，3个通道
    texts = torch.randint(0, 1000, (10, 1024)).to('cuda')  # 320个文本，每个文本1024个token
    targets = torch.randint(0, 1000, (10, 1024)).to('cuda')  # 目标输出

    # 创建数据集和数据加载器
    dataset = MultiModalDataset(images, texts, targets)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_images, batch_texts, batch_targets in dataloader:
            optimizer.zero_grad()

            output = model(batch_images, batch_texts)
            print(output.shape)
            print(batch_targets.shape)
            loss = next_token_loss(output, batch_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('batch',loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader)}')

    print("训练完成")