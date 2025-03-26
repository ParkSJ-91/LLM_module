import math
import copy
from collections import namedtuple
from typing import Optional, Any, Union, Callable, Tuple

from einops import rearrange

import bitsandbytes as bnb

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Embedding, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm, LogSoftmax
from torch.nn.init import xavier_uniform_

from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
import LLM_sample_module 

def get_kl_weight(step:int, kl_cycle_length: int, kl_ramp_length:int) -> float:
    cycle_pos = step % kl_cycle_length
    return min(1.0, cycle_pos / kl_ramp_length)

def get_cyclic_kl_beta(step: int, cycle_length: int) -> float:
    cycle_pos = step % cycle_length
    if cycle_pos < 0.5 * cycle_length:
        beta = 0.0
    elif cycle_pos < 0.75 * cycle_length:
        beta = (cycle_pos - 0.5 * cycle_length) / (0.25 * cycle_length)
    else:
        beta = 1.0
    return beta

class TransformerVAE(Module):
    def __init__(self, ntoken: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                       dim_feedforward: int = 2048, dropout: float = 0.1, d_latent: int = 128,
                       batch_first: bool = False, norm_first: bool = False,
                       layer_norm_eps: float = 1e-5):
        super().__init__()
        self.embedding = Embedding(ntoken, d_model)

        self.encoder = LLM_sample_module.TransformerLM(ntoken, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                                       dim_feedforward=dim_feedforward, dropout=dropout, causal=False,
                                                       batch_first=batch_first, norm_first=norm_first, flash=True)

        decoder_layer = FlashLatentInjectionLayer(d_model, nhead, dim_feedforward, d_model, dropout, batch_first=batch_first, norm_first=norm_first, **factory_kwargs) # latent already was already recovered its dimension
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = LLM_sample_module.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.fc_mu = Linear(d_model, d_latent)
        self.fc_logvar = Linear(d_model, d_latent)
        self.fc_decode = Linear(d_latent, d_model)

    def forward(self, data_input: torch.Tensor):

        emb = self.embedding(data_input) * math.sqrt(self.encoder.d_model)
        emb_dec = emb[:, 1:-1, :] # without cls token and last token

        encoder_out = self.encoder.decoder(emb,
                                           mask=self.encoder.alibi_slopes.to(device=data_input.device),
                                           tgt_is_causal=False)
        h_cls = encoder_out[:, 0, :]

        mu = self.fc_mu(h_cls)
        logvar = self.fc_logvar(h_cls)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z_recovered = self.fc_decode(z)
        dec_out = self.decoder(emb_dec, z_recovered,
                               mask=self.encoder.alibi_slopes.to(device=data_input.device), 
                               tgt_is_causal=True)
        return dec_out, mu, logvar

class FlashLatentInjectionLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, latent_dim: int = None, 
               dropout: float = 0.1, attention_dropout: float = 0.0,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
               device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.attn = FlashAttentionLatentInjection(d_model, nhead, latent_dim, dropout = dropout, attention_dropout = attention_dropout, layer_norm_eps = layer_norm_eps, batch_first = batch_first, norm_first = norm_first, **factory_kwargs)
        self.ff = FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, **factory_kwargs)

        self.norm_first = norm_first

    def forward(self, tgt, latent, mask=None, tgt_is_causal=True):#, tgt_key_padding_mask: Optional[Tensor] = None,
        x = tgt
        x = self.ff(self.attn(x, latent, mask=mask, tgt_is_causal=tgt_is_causal))
        return x

class FlashAttentionLatentInjection(Module):
    def __init__(self, d_model, nhead, latent_dim, dropout=0.1, attention_dropout=0.0, layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.device = device
        self.nhead = nhead

        self.head_dim = d_model // (nhead)
        self.c_attn = Linear(d_model, 3 * d_model, **factory_kwargs)
        self.c_proj = Linear(d_model, d_model, **factory_kwargs)

        self.flash_attn_func = flash_attn_func #flash_attn_qkvpacked_func

        self.latent_key_proj = Linear(latent_dim, d_model)
        self.latent_value_proj = Linear(latent_dim, d_model)

        self.norm_first = norm_first
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.attention_dropout = attention_dropout

    def forward(self, tgt, latent, mask=None, tgt_is_causal=True):
        x = tgt
        if self.norm_first:
            x = x + self._fa_block(self.norm(x), latent, mask, tgt_is_causal)
        else:
            x = self.norm(x + self._fa_block(x, latent, tgt_mask, tgt_key_padding_mask, rel_pos, tgt_is_causal))
        return x

    def _fa_block(self, x, latent, mask, is_causal=True, **kwargs):
        B, T, _ = x.shape

        query_projected = self.c_attn(x)
        
        qkv = query_projected.view(B, T, 3, self.nhead, self.head_dim)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        #print(latent.size())
        latent_k = self.latent_key_proj(latent).view(B, self.nhead, self.head_dim)
        latent_v = self.latent_value_proj(latent).view(B, self.nhead, self.head_dim)

        latent_k = latent_k.unsqueeze(1)
        latent_v = latent_v.unsqueeze(1)

        k = torch.cat([latent_k, k], dim=1)
        v = torch.cat([latent_v, v], dim=1)

        y = self.flash_attn_func(q,k,v,
                    dropout_p = self.attention_dropout,
                    causal = is_causal,
                    alibi_slopes = mask)
        y = y.view(B, T, self.nhead * self.head_dim)
        if y.isnan().any().item():
            exit()
        return self.dropout(self.c_proj(y))        

