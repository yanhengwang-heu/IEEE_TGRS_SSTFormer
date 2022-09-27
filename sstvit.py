import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward, CrossAttention, SSTransformer
import numpy as np

class SSTTransformerEncoder(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, cross_attn_depth=3, cross_attn_heads=8, dropout = 0):
        super().__init__()

        self.transformer = SSTransformer(dim, depth, heads, dim_head, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, num_patches, dropout)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(PreNorm(b_dim, CrossAttention(b_dim, heads = cross_attn_heads, dim_head=dim_head, dropout=0)))

    def forward(self, x1, x2):
        x1 = self.transformer(x1)
        x2 = self.transformer(x2)

        for cross_attn in self.cross_attn_layers:
            x1_class = x1[:, 0]
            x1 = x1[:, 1:]
            x2_class = x2[:, 0]
            x2 = x2[:, 1:]

            # Cross Attn
            cat1_q = x1_class.unsqueeze(1)
            cat1_qkv = torch.cat((cat1_q, x2), dim=1)
            cat1_out = cat1_q+cross_attn(cat1_qkv)
            x1 = torch.cat((cat1_out, x1), dim=1)
            cat2_q = x2_class.unsqueeze(1)
            cat2_qkv = torch.cat((cat2_q, x1), dim=1)
            cat2_out = cat2_q+cross_attn(cat2_qkv)
            x2 = torch.cat((cat2_out, x2), dim=1)

        return cat1_out, cat2_out

class SSTViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., multi_scale_enc_depth=1):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        self.num_patches = num_patches+1
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token_t1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_t2 = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.multi_scale_transformers = nn.ModuleList([])
        for _ in range(multi_scale_enc_depth):
            self.multi_scale_transformers.append(SSTTransformerEncoder(dim, depth, heads, dim_head, mlp_dim,b_dim, b_depth, b_heads, b_dim_head, b_mlp_head, self.num_patches,
                                                                                    dropout = 0.))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(b_dim),
            nn.Linear(b_dim, num_classes)
        )
    def forward(self, x1, x2):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')
        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x1 = self.patch_to_embedding(x1) #[b,n,dim]
        x2 = self.patch_to_embedding(x2)
        b, n, _ = x1.shape
        # add position embedding
        cls_tokens_t1 = repeat(self.cls_token_t1, '() n d -> b n d', b = b) #[b,1,dim]
        cls_tokens_t2 = repeat(self.cls_token_t2, '() n d -> b n d', b = b)

        x1 = torch.cat((cls_tokens_t1, x1), dim = 1) #[b,n+1,dim]
        x1 += self.pos_embedding[:, :(n + 1)]
        x1 = self.dropout(x1)
        x2 = torch.cat((cls_tokens_t2, x2), dim = 1) #[b,n+1,dim]
        x2 += self.pos_embedding[:, :(n + 1)]
        x2 = self.dropout(x2)
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        for multi_scale_transformer in self.multi_scale_transformers:
            out1, out2 = multi_scale_transformer(x1, x2)
        # classification: using cls_token output
        out1 = self.to_latent(out1[:,0])
        out2 = self.to_latent(out2[:,0])
        out = out1+out2
        # MLP classification layer
        return self.mlp_head(out)


