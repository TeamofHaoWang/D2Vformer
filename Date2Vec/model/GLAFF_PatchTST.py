'''
# TODO: Experiment Objective
# Test the ability of D2V_Fourier as position.
# Compare D2V, T2V, and position-based approaches.
# Use these position encodings and input them into PatchTST.
'''

from torch import nn
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from plugin.Plugin.model import Plugin

plt.rcParams['font.sans-serif'] = ['SimHei']  # To display Chinese labels correctly
plt.rcParams['axes.unicode_minus'] = False  # To display negative signs correctly

class GLAFF_PatchTST(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # Load parameters from configs
        c_in = configs.d_feature
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.pred_len = target_window

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        patch_len = configs.patch_len
        stride = configs.stride

        # Initialize PatchTST backbone model
        self.model = PatchTST_backbone(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=None,
            d_v=None,
            d_ff=d_ff,
            norm='BatchNorm',
            attn_dropout=False,
            dropout=dropout,
            act='gelu',
            key_padding_mask='auto',
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe='zeros',
            learn_pe=True,
            fc_dropout=dropout,
            head_dropout=False,
            padding_patch='end',
            pretrain_head=False,
            head_type='flatten',
            individual=False,
            revin=True,
            affine=False,
            subtract_last=False,
            verbose=False
        )

        # Initialize Plugin module
        self.plugin = Plugin(configs, c_in)

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark, mode):
        """
        Forward pass through the model.

        Args:
            batch_x (torch.Tensor): Input tensor of shape [Batch, Input length, Channel]
            batch_x_mark (torch.Tensor): Time features for input
            batch_y (torch.Tensor): Target tensor for prediction
            batch_y_mark (torch.Tensor): Time features for target
            mode (str): Mode of operation (e.g., 'train', 'test')

        Returns:
            pred (torch.Tensor): The predicted output tensor
        """
        # Make copies of the input for later use
        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = batch_x.clone(), batch_x_mark.clone(), batch_y_mark.clone()

        # Transform input data shape for PatchTST
        x = batch_x.permute(0, 2, 1)  # Change shape to [Batch, Channel, Input length]
        x = self.model(x)  # Pass input through the PatchTST backbone
        x = x.permute(0, 2, 1)  # Change shape back to [Batch, Input length, Channel]

        # Pass through Plugin module for final prediction
        pred = self.plugin(x_enc_copy, x_mark_enc_copy, x, x_mark_dec_copy[:, -self.pred_len:, :])

        return pred  # Return the predicted tensor
