from torch.utils.data import DataLoader
from sft_dataset import InstructionDataset
import pickle

def load(instructions,reponses,tokenizer,batch_size,max_length=512):
    dataset = InstructionDataset(instructions,reponses, tokenizer,max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

if __name__ == '__main__':
    with open('../output/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    itros = [
        'whats this?',
        'whats that?'
    ]
    re =[
        "This section shows several tokenizer algorithms",
        "Hopefully, you will be able to understand how they are trained and generate tokens",
    ]
    max_length = 512
    data = load(itros,re,max_length,4)
