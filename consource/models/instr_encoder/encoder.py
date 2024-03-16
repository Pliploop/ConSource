import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder
        
    def forward(self, x):
        return self.encoder(x)