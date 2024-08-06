from collections import defaultdict
from pprint import pprint

class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = ['[PAD]','[END]']
        self.splits = {}
        self.merges = {}

    def compute_pair_freqs(self, splits):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def train(self,corpus):
        # 词频统计
        for text in corpus:
            words = text.split()
            for word in words:
                self.word_freqs[word] += 1
        print('frequence counted finished')

        # 初始化词表
        for word in self.word_freqs.keys():
            print('voc prepared..')
            for letter in word:
                if letter not in self.vocab:
                    self.vocab.append(letter)
        self.vocab.append("</w>")
        print('voc initialized')

        # 初始化分割
        self.splits = {word: list(word) + ['</w>'] for word in self.word_freqs.keys()}
        print('splits initialized')
        # 训练过程
        while len(self.vocab) < self.vocab_size:
            print('continue training,wait')
            pair_freqs = self.compute_pair_freqs(self.splits)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.splits = self.merge_pair(*best_pair, self.splits)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])

    def tokenize(self, text):
        splits = [list(word) + ['</w>'] for word in text.split()]
        for pair, merge in self.merges.items():
            print('finish')
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.index(token) for token in tokens]

    def decode(self, encoded):
        tokens = [self.vocab[index] for index in encoded]
        return ' '.join(tokens).replace(' </w>', '').replace('</w>', '')

# 示例使用
if __name__ == '__main__':
    corpus = [
        "一蓑烟雨任平生，爷傲奈我何",
        "太阳花儿笑江湖，踏遍青山人未老",
        "This section shows several tokenizer algorithms",
        "Hopefully, you will be able to understand how they are trained and generate tokens",
    ]

    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.train(corpus)

    # 编码文本
    encoded = tokenizer.encode("笑傲江湖")
    print(encoded)
    print(tokenizer.decode(encoded))
