
from torch import nn
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from layers.Date2Vec import Date2Vec_Fourier



class D2V_Fourier_PatchTST(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # load parameters
        c_in = configs.d_feature
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        patch_len = configs.patch_len
        stride = configs.stride
        self.mark_index = configs.mark_index

        self.model = PatchTST_backbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                       patch_len=patch_len, stride=stride,n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm',
                                       attn_dropout=False,dropout=dropout, act='gelu', key_padding_mask='auto',
                                       padding_var=None,attn_mask=None, res_attention=True, pre_norm=False,
                                       store_attn=False,pe='zeros', learn_pe=True, fc_dropout=dropout, head_dropout=False,
                                       padding_patch='end',pretrain_head=False, head_type='flatten', individual=False,
                                       revin=True, affine=False,subtract_last=False, verbose=False)


        # Date2Vec
        self.D2V_position = Date2Vec_Fourier(context_window, configs.T2V_outmodel, c_in,
                                         d_mark=len(self.mark_index), save_path=configs.output_path)

    def forward(self, batch_x, batch_x_mark,batch_y, batch_y_mark,mode):
        # batch_x: [Batch, Input length, Channel]

        x = batch_x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        batch_x_mark = batch_x_mark[:, :, self.mark_index] # TODO 针对不同数据集进行mark选取
        D2V_output = self.D2V_position(x,batch_x_mark.permute(0,2,1), mode)
        x = x + torch.sum(D2V_output,dim=-1)
        x = self.model(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]

        return x
