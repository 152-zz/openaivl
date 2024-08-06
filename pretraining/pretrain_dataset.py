
from torch.utils.data import Dataset
import torch

class SentenceDataset(Dataset):
    #sentences[sentence1,sentence2...]
    def __init__(self, sentences, tokenizer, max_length=512, padding_value=0):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = padding_value
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for sentence in self.sentences:
            # Tokenize the sentence
            encoded = self.tokenizer.encode(sentence)
            input_seq = torch.tensor(encoded)
            input_seq = self._pad_sequence(input_seq)
            label_seq = input_seq.clone()

            # Shift the label sequence by one position
            label_seq = torch.cat((label_seq[1:],torch.tensor([1])))

            data.append((input_seq, label_seq))
        return data

    def _pad_sequence(self, sequence):
        # 确保序列长度为 max_length，不足时填充
        if len(sequence) < self.max_length-1:
            padding = torch.tensor([self.padding_value] * (self.max_length - 1-len(sequence)))
            sequence = torch.cat((sequence, torch.tensor([1])))
            sequence = torch.cat((sequence, padding))
        elif len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq, label_seq = self.data[idx]
        return {
            'input_ids': input_seq,
            'labels': label_seq
        }
