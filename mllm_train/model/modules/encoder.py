import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.rope import RoPE
from model.modules.RMSNorm import RMSNorm
from model.modules.swiglu import FFNSWIGLU
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

        query, key = RoPE(query, key)

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
        self.RMS1 = RMSNorm(embed_size)
        self.RMS2 = RMSNorm(embed_size)
        self.feed_forward = FFNSWIGLU(embed_size,forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        value = self.RMS1(value)
        key = self.RMS1(key)
        query = self.RMS1(value)

        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.RMS2(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(forward + x)
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
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

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
        out = self.word_embedding(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out