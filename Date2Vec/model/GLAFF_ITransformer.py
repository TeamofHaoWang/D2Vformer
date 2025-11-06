import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.iTransformer_EncDec import Encoder, EncoderLayer
from layers.iTransformer_SelfAtt_Famliy import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from plugin.Plugin.model import Plugin

import numpy as np

# class Date2Vec_Fourier(nn.Module):
#     def __init__(self, in_features, out_features, d_features, d_mark):
#         # For D2V, here the input in_features refers to seq_len and
#         # out_features refers to how many sinusoids are used to characterise the corresponding each D
#         super(Date2Vec_Fourier, self).__init__()
#         self.out_features = out_features

#         self.f = torch.cos
#         self.dominance_freq = int(in_features // 24 + 1) * 2 + 10
#         self.freq_upsampler_real = nn.Linear(self.dominance_freq,
#                                              out_features)  # complex layer for frequency upcampling]
#         self.freq_upsampler_imag = nn.Linear(self.dominance_freq,
#                                              out_features)  # complex layer for frequency upcampling]
#         # nn.init.uniform_(self.freq_upsampler_real.weight, 0, 1)
#         # nn.init.uniform_(self.freq_upsampler_imag.weight, 0, 1)

#         self.w_fourier = torch.arange(0, 2 * torch.pi, 2 * torch.pi / out_features)
#         print('Using Date2Vec_Fourier')

#     def forward(self, data, tau):
#         low_specx = torch.fft.rfft(data, dim=-1)
#         if low_specx.shape[1] < self.dominance_freq:
#             low_specx = torch.fft.rfft(data, n=2 * (self.dominance_freq - 1), dim=-1)  # n 表示傅里叶域的变换长度

#         low_specx = torch.view_as_real(low_specx[:, :, 0:self.dominance_freq])  # 复数值 按 real值存储变成2维向量
#         low_specx_real = low_specx[:, :, :, 0]
#         low_specx_imag = low_specx[:, :, :, 1]

#         real = self.freq_upsampler_real(low_specx_real)
#         imag = self.freq_upsampler_imag(low_specx_imag)
#         low_specxy_real = real - imag
#         low_specxy_imag = real + imag

#         # 相位角 = arctan(imag/real)  幅值= sqrt(real^2 + imag^2)
#         attitude = torch.sqrt(torch.pow(low_specxy_real, 2) + torch.pow(low_specxy_imag, 2))
#         theta = torch.atan2(low_specxy_imag, low_specxy_real)

#         output = self.D2V(attitude, theta, tau, self.f)
#         return output

#     def D2V(self, attitude, theta, tau, f):
#         _, D, _ = attitude.shape
#         tau = tau[:, 0, :].unsqueeze(-1).repeat(1, 1, D)  # (B,L+O,d_feature)
#         tau = tau.transpose(1, -1)  # (B,d_feature,L+O)
#         theta = theta.unsqueeze(2)
#         attitude = attitude.unsqueeze(2)
#         tau = tau.unsqueeze(2).transpose(2, -1)

#         w_fourier = self.w_fourier.unsqueeze(-1).to(tau.device)
#         w_tau = torch.einsum('bdln,fn->bdlf', tau, w_fourier)
#         v1 = attitude * f(w_tau + theta)
#         return v1



class GLAFF_iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(GLAFF_iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.use_norm = True
        # Embedding_original_iTransformer
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        #self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.plugin = Plugin(configs, configs.d_feature)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        # position = self.D2V_position(x_enc.permute(0,2,1),x_mark_enc.permute(0,2,1))
        # x_enc = x_enc + torch.sum(position,dim=-1).permute(0,2,1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  #这里只是对输入的特征维度进行一个线性映射 covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mode):
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        pred = dec_out[:, -self.pred_len:, :]
        pred = self.plugin(x_enc_copy, x_mark_enc_copy, pred, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred[:, -self.pred_len:, :]  # [B, L, D]