import math
import copy
from typing import Optional, Any, Union, Callable, Tuple

from einops import rearrange

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, Embedding, MultiheadAttention, ModuleList, Dropout, Linear, LayerNorm, LogSoftmax
from torch.nn.init import xavier_uniform_

from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

class BERTEmbedding(Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()

        self.embed_size = embed_size 

        self.token = Embedding(vocab_size, embed_size, padding_idx = 0)
        self.segment = Embedding(3, embed_size, padding_idx = 0)
       
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.segment(segment_label)
        return x

class NextSentencePrediction(Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = Linear(hidden, 2)
        self.softmax = LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = Linear(hidden, vocab_size)
        self.softmax = LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(Module):
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)

class TransformerLM(Module):
    def __init__(self, ntoken: int, d_model: int = 512, nhead: int = 8, 
               num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
               attention_dropout: float = 0.0, layer_dropout: float = 0.,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, flash: bool = False,
               alibi_pos_bias: bool = True, alibi_nhead: int = None, 
               layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
               max_seq_len: int = 1024, causal: bool = True) -> None:
        super().__init__()

        self.model_type = 'Transformer'
        self.nhead = nhead
        self.max_seq_len = max_seq_len
        self.is_causal = causal

        self.flash = flash

        # alibi
        if not alibi_nhead:
            alibi_nhead = nhead
            if not flash:
                alibi_pos_klass = AlibiPositionalBias
                self.rel_pos = alibi_pos_klass(heads = alibi_nhead, total_heads = nhead)
            else:
                self.alibi_slopes = get_Alibi_slopes(alibi_nhead)

        if flash:
            decoder_layer = FlashDecoderLayer(d_model, nhead, dim_feedforward, dropout, attention_dropout, activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        if self.is_causal:
            self.embedding = Embedding(ntoken, d_model)
        else:
            self.embedding = BERTEmbedding(ntoken, d_model)
        self.d_model = d_model
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.linear = Linear(d_model, ntoken)

        self._reset_parameters()

        self.batch_first = batch_first

    def forward(self, tgt: Tensor, segment_info: Optional[Tensor] = None) -> Tensor:

        is_batches = tgt.dim() == 3

        if not self.flash:
            tgt_key_padding_mask, causal_mask = create_mask(tgt)
            if tgt_key_padding_mask is not None:
                mask = rearrange(tgt_key_padding_mask, 'b j -> b 1 1 j')
                attn_bias = self.rel_pos(tgt.shape[1], tgt.device, tgt.dtype)
                attn_bias = rearrange(attn_bias, 'h 1 j -> 1 h 1 j')

                mask = attn_bias
        else:
            mask = self.alibi_slopes.to(device=tgt.device)

        if self.is_causal:
            tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        else:
            tgt = self.embedding(tgt, segment_info) * math.sqrt(self.d_model)

        output = self.decoder(tgt,
                      mask=mask,
                      tgt_is_causal = self.is_causal
                      )
        return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def create_mask(tgt: Tensor, tgt_pad_idx: int = 0) -> Tuple[Tensor, Tensor]:
    padding_mask = _create_padding_mask(tgt, tgt_pad_idx)
    nopeak_mask = _create_nopeak_mask(tgt)

    return padding_mask, nopeak_mask


class PositionalEncoding(Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class AlibiPositionalBias(Module):
    def __init__(self, heads, total_heads):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

    def get_bias(self, seqLen):
        return torch.arange(1-seqLen, 1, dtype=torch.int32).view(1, 1, seqLen)
    def get_slopes(self, n_heads, alibi_bias_max = 8):
        _n_heads = 2**math.ceil(math.log2(n_heads))
        m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
        m = m.mul(alibi_bias_max / _n_heads)
        slopes = (1. / torch.pow(2, m))

        if _n_heads != n_heads:
            # if n_heads is not a power of two,
            # Huggingface and FasterTransformer calculate slopes normally,
            # then return this strided concatenation of slopes
            slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]

        return slopes

    def forward(self, seqLen, device, dtype):
        slopes = self.get_slopes(self.total_heads).to(device=device)
        bias = self.get_bias(seqLen).to(device=device)
        self.bias = bias * slopes
        return self.bias.to(dtype=dtype)
        
def get_Alibi_slopes(n_heads, alibi_bias_max = 8):
    _n_heads = 2**math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = (1. / torch.pow(2, m))

    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, mask: Optional[Tensor] = None, tgt_is_causal: bool = True) -> Tensor:

        output = tgt
        
        for mod in self.layers:
            output = mod(output, mask=mask, 
                     tgt_is_causal=tgt_is_causal
                     )

        if self.norm is not None:    
            output = self.norm(output)
        return output

class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # for Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
              tgt_is_causal: bool = True) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = True) -> Tensor:
        x = self.self_attn(x, x, x, 
                   attn_mask=attn_mask,
                   key_padding_mask=key_padding_mask,
                   is_causal=is_causal,
                   need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class FlashDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
               attention_dropout: float = 0.0, 
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False) -> None:
        super().__init__()

        self.attn = FlashAttention(d_model, nhead, dropout = dropout, attention_dropout = attention_dropout, layer_norm_eps = layer_norm_eps, batch_first = batch_first, norm_first = norm_first)
        self.ff = FeedForward(d_model, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first)

        self.norm_first = norm_first

    def forward(self, tgt: Tensor, mask: Optional[Tensor] = None, tgt_is_causal: bool = True) -> Tensor:
        x = tgt
        x = self.ff(self.attn(x, mask=mask, tgt_is_causal=tgt_is_causal))
        return x

class FlashAttention(Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, attention_dropout: float = 0.0, layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False):
        super().__init__()

        self.create_causal_mask = create_causal_mask
        self.nhead = nhead
        self.c_attn = Linear(d_model, 3 * d_model)
        self.c_proj = Linear(d_model, d_model)

        self.flash_attn_func = flash_attn_qkvpacked_func
                
        
        self.norm_first = norm_first
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = Dropout(dropout)
        self.attention_dropout = attention_dropout

    def forward(self, tgt: Tensor, mask: Optional[Tensor] = None, tgt_is_causal: bool = True) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._fa_block(self.norm(x), mask, tgt_is_causal)
        else:
            x = self.norm(x + self._fa_block(x, tgt_mask, tgt_key_padding_mask, rel_pos, tgt_is_causal))
        return x

    def _fa_block(self, x: Tensor, mask: Optional[Tensor], is_causal: bool = True, **kwargs) -> Tensor:
        query_projected = self.c_attn(x)
        batch_size = query_projected.size(0)
        seq_len = query_projected.size(1)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.nhead * 3)
        
        qkv = query_projected.view(batch_size, seq_len, 3, self.nhead, head_dim)
        y = self.flash_attn_func(qkv,
                    dropout_p = self.attention_dropout,
                    causal = is_causal,
                    alibi_slopes = mask)
        y = y.view(batch_size, seq_len, self.nhead * head_dim)
        if y.isnan().any().item():
            exit()
        return self.dropout(self.c_proj(y))

class FeedForward(Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5, norm_first: bool = False) -> None:
        super().__init__()

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm = LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation
        
        self.norm_first = norm_first
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.norm_first:
            x = x + self._ff_block(self.norm(x))
        else:
            x = self.norm(x + self._ff_block(x))
        return x
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _create_padding_mask(seq: Tensor, pad_idx: int = 0) -> Tensor:
    return ~(seq == pad_idx)#.unsqueeze(-2)

def _create_nopeak_mask(tgt) -> Tensor:
    batch_size, seq_len = tgt.size()
    nopeak_mask = (torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1)).bool()
    return nopeak_mask

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j-i+1).to(device)

