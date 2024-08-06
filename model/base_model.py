import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension needs to be divisible by number of heads"

        # 线性层用于生成查询、键和值
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # 输出线性层用于合并多头的输出
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 生成查询、键、值
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # 分割成多个头
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        # 加权求和
        out = torch.matmul(attention, value)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 输出线性层
        out = self.out_linear(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class bumblebee(nn.Module):
        def __init__(
                self,
                src_vocab_size,
                embed_size,
                num_layers,
                heads,
                forward_expansion,
                dropout,
                device,
                max_length,
        ):
            super(bumblebee, self).__init__()
            self.encoder = Encoder(
                src_vocab_size,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                max_length,
            )
            self.decoder = nn.Linear(embed_size,src_vocab_size)
            self.softmax = nn.Softmax(dim=-1)
            self.device = device

        def forward(self, x, mask):
            enc_src = self.encoder(x, mask)
            out = self.decoder(enc_src)
            #out = self.softmax(self.decoder(enc_src))
            return out

def create_look_ahead_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    return mask

if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab_size = 1000
    embed_size = 256
    num_layers = 1
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 100

    model = bumblebee(
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ).to(device)


    mask = create_look_ahead_mask(max_length).to(device)

    # Dummy input
    src = torch.randint(0, src_vocab_size, (64, max_length)).to(device)

    output = model(src, mask)
    print(output)
    #print(output.shape)  # Should be [64, max_length, embed_size]
