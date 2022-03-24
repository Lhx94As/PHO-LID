import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class oldPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, features_dim, device):
        super(oldPositionalEncoding, self).__init__()
        pos_enc = np.array(
            [[pos/np.power(10000, 2.0*(i//2)/features_dim) for i in range(features_dim)]
             for pos in range(max_seq_len)])
        pos_enc[:,0::2] = np.sin(pos_enc[:,0::2])
        pos_enc[:,1::2] = np.cos(pos_enc[:,1::2])
        self.pos_enc = torch.from_numpy(pos_enc).to(device)

    def forward(self, x, seq_len):
        # x: [B, T, feat_dim]
        for i in range(x.size(0)):
            len_ = seq_len[i]
            x[i,:len_,:] += self.pos_enc[:len_, :]
        return x

class PositionalEncoding(nn.Module):
    """
    PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
    PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, max_seq_len, features_dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, features_dim, requires_grad=False)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, features_dim, 2).float() * -(math.log(10000.0) / features_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, seq_len):
        for i in range(x.size(0)):
            len_ = seq_len[i]
            x[i, :len_, :] += self.pe[:len_, :]
        # print(x.size())
        return x




class LayerNorm(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNorm, self).__init__()
        # d_hid = feat_dim
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True,)
        std = x.std(dim=-1, keepdim=True,)
        ln_out = (x - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta
        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, atten_mask=None):
        # queries: [B, n_head, len_queries, d_k]
        # keys: [B, n_head, len_keys, d_k]
        # values: [B, n_head, len_values, d_v] note: len_keys = len_values
        # print(q.size())
        scores = torch.matmul(q, k.transpose(-1, -2))/ self.scale_factor
        if atten_mask is not None:
            # print(atten_mask.size(),scores.size())
            assert atten_mask.size() == scores.size()
            scores.masked_fill_(atten_mask, -1e9)
        atten = self.dropout(self.softmax(scores))
        context = torch.matmul(atten, v)
        return context, atten


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight) #For Sigmoid
        # init.kaiming_normal_(self.linear.weight) #for ReLU
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(self.d_model, d_k*n_heads)
        self.w_k = Linear(self.d_model, d_k*n_heads)
        self.w_v = Linear(self.d_model, d_v*n_heads)

        self.attenion = ScaledDotProductAttention(d_k=d_k, dropout=dropout)

    def forward(self, x, atten_mask):
        batch_size = x.size(0)

        q_ = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_ = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_ = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # q_: [Batch, n_heads, len, d_k]
        # k_: [Batch, n_heads, len, d_k]
        # v_: [Batch, n_heads, len, d_v]
        if atten_mask is not None:
            # [Batch, len, len] -> [Batch, n_heads, len, len]
            atten_mask = atten_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, atten = self.attenion(q_, k_, v_, atten_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads*self.d_v)
        return context, atten

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.multihead_attention = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.linear = Linear(n_heads*d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)
    def forward(self, x, atten_mask):
        # x_: [Batch, n_heads, len, feat_dim]
        residual = x
        context, atten = self.multihead_attention(x, atten_mask)
        output = self.dropout(self.linear(context))
        output = self.layernorm(output + residual)
        # output: [Batch, len, feat_dim]
        return output, atten

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        # x = self.layernorm(x) #pre-LN
        output = self.relu(self.fc1(x))
        output = self.dropout(self.fc2(output))
        output = self.layernorm(output+residual) #post-LN
        # output = output+residual #pre-LN
        return output

class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttentionLayer(d_model, d_k, d_v, n_heads, dropout)
        self.position_wise_ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    def forward(self, x, atten_mask):
        enc_output, atten = self.self_attention(x, atten_mask)
        enc_output = self.position_wise_ff(enc_output)
        return enc_output, atten


