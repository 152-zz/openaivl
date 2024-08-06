import sys
sys.path.append('../model')
from clip import CLIPModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def train_clip(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        ep_loss = torch.tensor(0.0)
        for images, texts in dataloader:
            optimizer.zero_grad()
            image_embeddings, text_embeddings = model(images, texts)
            loss = contrastive_loss(image_embeddings, text_embeddings)
            loss.backward()
            optimizer.step()
            ep_loss += loss
        print(f"Epoch {epoch+1}, Loss: {ep_loss}")

if __name__ == '__main__':
    model = CLIPModel()
    #数据样例
    image = torch.randn(10,3,224,224)
    text = torch.randint(0,500,(10,50))

    #数据加载
    dataset = TensorDataset(image, text)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    #优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    train_clip(model,dataloader,optimizer,epochs = 150)









