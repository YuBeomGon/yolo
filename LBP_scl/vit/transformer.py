# this code is from https://github.com/FrancescoSaverioZuppichini/ViT, tuned

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, transforms
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from config import *

# class FeatureExtraction(nn.Module) :
#     def __init__(self) :
        

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = EMBDED_SIZE):
        super().__init__()
        self.image_size = 1568
        self.patch_size = patch_size
        self.patch_size_2nd = 8
        self.stride_2nd = 6
#         self.projection = FeatureExtraction()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels=in_channels, out_channels=int(emb_size/6), 
                      kernel_size=patch_size, stride=patch_size),
            nn.Conv2d(in_channels=int(emb_size/6), out_channels=emb_size, 
                      kernel_size=self.patch_size_2nd, stride=self.stride_2nd), #128 patch size, 32 stride
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # cls token for patch image is not useless
#         self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn(((self.image_size - patch_size*self.patch_size_2nd) \
                                                   // (patch_size*self.stride_2nd) +1 ) **2,
                                                  emb_size))
                
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
#         cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
#         x = torch.cat([cls_tokens, x], dim=1)   
        x += self.positions
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = EMBDED_SIZE, num_heads: int = NUM_HEADS, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
#         self.keys = nn.Linear(emb_size, emb_size)
#         self.queries = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, num_heads * 3)
        self.queries = nn.Linear(emb_size, num_heads * 3)        
        self.values = nn.Linear(emb_size, emb_size)
#         self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
#         self.scaling = (self.emb_size // num_heads) ** -0.5
        self.scaling = (3) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
#         # split keys, queries and values in num_heads
#         qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
#         queries, keys, values = qkv[0], qkv[1], qkv[2]
#         # sum up over the last axis
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)        
            
#         energy /= self.scaling       
        energy *= self.scaling       
#         att = F.softmax(energy, dim=-1) * self.scaling
        att = F.softmax(energy, dim=-1)
        if self.att_drop is not None:
            att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)    
    
# class BertLayer(nn.Sequential):
class BertLayer(nn.Module):    
    def __init__(self,
                 emb_size: int = EMBDED_SIZE,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__()
        self.mlp = FeedForwardBlock(emb_size, forward_expansion, drop_p)
        self.mha = MultiHeadAttention(emb_size=EMBDED_SIZE, num_heads=NUM_HEADS, dropout=0)
        self.layernorm_mlp = nn.LayerNorm(emb_size)
        self.layernorm_mha = nn.LayerNorm(emb_size)
        self.dropout_mlp = nn.Dropout(drop_p)
        self.dropout_mha = nn.Dropout(drop_p)
        
    def forward(self, x: Tensor) -> Tensor:
        skipped = x
        x = self.layernorm_mha(x)
        x = self.mha(x)
        x = self.dropout_mha(x)
        x += skipped
        
        skipped = x
        x = self.layernorm_mlp(x)
        x = self.mlp(x)
        x = self.dropout_mlp(x)
        x += skipped
        
        return x
    
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer() for _ in range(config.num_hidden_layers)])
        self.embeding = PatchEmbedding()
        self.output = nn.Linear(config.emb_size, config.Num_class) # 1 for objectness, 2 for class (normal cell or abnormal cell)
        
    def forward (self, x : Tensor) -> Tensor :
        x = self.embeding(x)
        for i, layer_module in enumerate(self.layer):
            x = layer_module(x)
                       
        return self.output(x)
        
            