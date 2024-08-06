from torch.utils.data import Dataset, DataLoader
class MultiModalDataset(Dataset):
    def __init__(self, images, texts, targets):
        self.images = images
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.targets[idx]