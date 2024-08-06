from torch.utils.data import DataLoader
from pretrain_dataset import SentenceDataset
import pickle

def load(sentences,tokenizer,batch_size,max_length=512):
    dataset = SentenceDataset(sentences, tokenizer,max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == '__main__':
    with open('../output/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    texts =[
        "This is the Hugging Face Course",
        "This chapter is about tokenization",
        "This section shows several tokenizer algorithms",
        "Hopefully, you will be able to understand how they are trained and generate tokens",
    ]
    max_length = 512
    data = load(texts,tokenizer,max_length,4)
