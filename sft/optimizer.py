import torch
from tqdm import tqdm
from model.base_model import create_look_ahead_mask

def train(model, train_loader, val_loader, criterion, num_epochs, device, max_length):
    mask = create_look_ahead_mask(max_length).to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 使用 tqdm 来包装 train_loader 以显示进度条
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True):
            input_ids = batch['input_ids'].to(device).long()
            labels = batch['labels'].to(device).long()

            outputs = model(input_ids, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            model.backward(loss)
            model.step()
            running_loss += loss.item()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device).long()
                labels = batch['labels'].to(device).long()

                outputs = model(input_ids, mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_running_loss += loss.item()

        # 打印每个epoch的平均损失
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        print(f'Validation Loss: {val_running_loss / len(val_loader)}')
    model_path = '../output/zzyllm_final.pth'
    torch.save(model.state_dict(), model_path)

