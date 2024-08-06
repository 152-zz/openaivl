import torch
import torch.nn as nn
from model.vit import VisionTransformer
from model.llama_model import bumblebee
import pickle


# 3. 定义一个多模态模型类
class MultiModalModel(torch.nn.Module):
    def __init__(self,emb_vit=400,emb_text=400):
        super(MultiModalModel, self).__init__()
        self.vit_model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, emb_size=emb_vit, depth=3, num_heads=8, mlp_ratio=2, dropout=0.1)
        self.language_model = bumblebee(src_vocab_size=5000,embed_size = emb_text,num_layers=3,heads=8,forward_expansion=2,dropout=0.1,device='cuda',max_length=1024)
        self.adapter = nn.Linear(in_features = emb_vit, out_features = emb_text)

    def forward(self, images, texts):
        # 处理图像
        with torch.no_grad():
            image_features = self.vit_model(images)
        image_features = self.adapter(image_features)
        image_seq_len = image_features.shape[1]
        # 处理文本
        text_features = self.language_model.encoder.word_embedding(texts)

        # 融合图像和文本特征
        combined_features = torch.cat([image_features, text_features], dim=1)

        # 通过语言模型的解码器部分
        for layer in self.language_model.encoder.layers:
            combined_features = layer(combined_features, combined_features, combined_features, mask=None)

        output = self.language_model.decoder(combined_features)
        output = output[:,image_seq_len:]

        return output

if __name__ == '__main__':

    emb_vit = 400
    emb_text = 400
    model = MultiModalModel(emb_vit, emb_text)

    with open('../../output/tokenizer1.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    images = torch.randn(1, 3, 224, 224)  # 示例图像数据
    text = "What is happening in the picture?"#示例文本数据
    text = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    print(text)

    output = model(images, text)

    print(output.shape)
