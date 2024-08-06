import torch
import torch.nn as nn
from model.modules.encoder import Encoder

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
    model = bumblebee(
        src_vocab_size=500,
        embed_size=16,
        num_layers=2,
        heads=8,
        forward_expansion=2,
        dropout=0.1,
        device='cuda',
        max_length=1000,
    )
    x = torch.randint(0,499,(2,500))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    y = model(x,mask = None)
