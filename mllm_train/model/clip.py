import torch
import torch.nn as nn
from model.vit import VisionTransformer as vittower
from model.modules.encoder import Encoder as texttower

class CLIPModel(nn.Module):
    def __init__(self,emb_vit=400,emb_text=400,projection_dim=300):
        super(CLIPModel, self).__init__()
        self.vision_model = vittower(img_size=224, patch_size=16, in_channels=3, emb_size=emb_vit, depth=8, num_heads=8, mlp_ratio=4, dropout=0.1)
        self.text_model = texttower(src_vocab_size=5000,embed_size = emb_text,num_layers=8,heads=8,forward_expansion=4,dropout=0.1,device='cuda')
        self.vision_projection = nn.Linear(emb_vit, projection_dim)
        self.text_projection = nn.Linear(emb_text, projection_dim)

    def forward(self, images, texts):
        image_features = self.vision_model(images)
        text_features = self.text_model(texts,mask = None)
        image_embeddings = self.vision_projection(image_features[:,1:].mean(dim=1))
        text_embeddings = self.text_projection(text_features.mean(dim=1))
        return image_embeddings, text_embeddings


def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

if __name__ == '__main__':
    model = CLIPModel()
    image = torch.randn(10,3,224,224)
    #print(image.shape)
    text = torch.randint(0,1000,(3,50))
    #print(text.shape)
    image_embed,text_embed = model(image,text)
    print(image_embed.shape,text_embed.shape)


