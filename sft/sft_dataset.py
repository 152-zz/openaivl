from torch.utils.data import Dataset
import torch

from torch.utils.data import Dataset
import torch
import pickle

#加载的监督微调数据需要成对，[instruction,instruction...]和[response,response...]
class InstructionDataset(Dataset):
    def __init__(self, instructions, responses, tokenizer, max_length=512, padding_value=0):
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = padding_value
        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        for instruction, response in zip(self.instructions, self.responses):
            # Tokenize the instruction and response
            encoded_instruction = self.tokenizer.encode(instruction)
            encoded_response = self.tokenizer.encode(response)

            # Combine instruction and response
            combined_seq = encoded_instruction + encoded_response +[torch.Tensor(1)]
            input_seq = torch.tensor(combined_seq)
            input_seq = self._pad_sequence(input_seq)

            # Create label sequence
            label_seq = input_seq.clone()
            label_seq[:len(encoded_instruction)] = self.padding_value  # Mask out the instruction part

            data.append((input_seq, label_seq))
        return data

    def _pad_sequence(self, sequence):
        # Ensure sequence length is max_length, pad if necessary
        if len(sequence) < self.max_length:
            padding = torch.tensor([self.padding_value] * (self.max_length - len(sequence)))
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

if __name__ == '__main__':
    instructions = ['whats this','whats that']
    responses = ['its this', 'its that']
    with open('../output/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    dataset = InstructionDataset(instructions,responses,tokenizer)
    print(dataset.data[0])