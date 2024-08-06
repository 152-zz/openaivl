import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.encoder import TransformerBlock
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=500, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_size=768, depth=12, num_heads=8, mlp_ratio=4, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, dropout,mlp_ratio) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x,x,x,None)
        #cls_token = x[:, 0]
        #x = cls_token
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = VisionTransformer()
    # 示例输入
    dummy_input = torch.randn(1, 3, 224, 224)
    y = model(dummy_input)
    print(y.shape)
