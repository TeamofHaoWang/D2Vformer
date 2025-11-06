import math
import torch
import torch.nn as nn
import os
import seaborn as sns
import matplotlib.pyplot as plt
from layers.Date2Vec import *

class Fusion_Block_GLAFF(nn.Module):
    def __init__(self, args, save_path):
        super(Fusion_Block_GLAFF, self).__init__()

        # Initialize layers and parameters
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()

        # Feedforward layers for feature transformation
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)

        # Normalization layers
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        self.norm3 = nn.BatchNorm1d(args.d_feature)

        # Output projection and linear layer
        self.fc_out = nn.Linear(args.d_model, args.d_feature, bias=True)
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out = nn.Linear(patch_num * args.patch_len, args.pred_len)

        # Visualization and hyperparameters
        self.save_path = save_path
        self.flag = True
        self.q = 0.75

        # MLP for weighting the components
        self.MLP = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model),
            nn.GELU(),
            nn.Linear(args.d_model, 2),
            nn.Softmax(dim=-1)
        )

    def visual_attention(self, A):
        """Visualize attention score heatmap"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tmp = A[0, 0, :, :].clone().detach().cpu().numpy()
        plt.figure()
        ax = sns.heatmap(tmp, annot=False, fmt=".2e", cmap='coolwarm')
        cbar = ax.collections[0].colorbar
        decimal_places = 3
        ticks = np.linspace(tmp.min(), tmp.max(), num=5)
        tick_labels = [f'{tick:.{decimal_places}f}' for tick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Attention Score')
        plt.savefig(os.path.join(self.save_path, '_Attention_Score'))
        return

    def forward(self, x, x_date, y_date, mode):
        """
        Forward pass of the Fusion Block GLAFF.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, P_N, P_L)
            x_date (torch.Tensor): Date features for input tensor of shape (B, L, D, K)
            y_date (torch.Tensor): Date features for output tensor of shape (B, O, D, K)
            mode (str): Mode of operation ('train' or 'test')

        Returns:
            pred (torch.Tensor): The final predicted tensor
        """
        B, L, D, k = x_date.shape
        x_enc_true = x  # Already normalized

        # Robust normalization for date features
        robust_means_true = torch.median(x_enc_true, dim=1, keepdim=True)[0]
        robust_means_map = torch.median(x_date, dim=1, keepdim=True)[0]
        robust_stdev_true = torch.quantile(x_enc_true, self.q, 1, True) - torch.quantile(x_enc_true, 1 - self.q, 1, True) + 1e-6
        robust_stdev_map = torch.quantile(x_date, self.q, 1, True) - torch.quantile(x_date, 1 - self.q, 1, True) + 1e-6
        x_date_map = (x_date - robust_means_map) / robust_stdev_map * robust_stdev_true.unsqueeze(-1) + robust_means_true.unsqueeze(-1)
        y_date_map = (y_date - robust_means_map) / robust_stdev_map * robust_stdev_true.unsqueeze(-1) + robust_means_true.unsqueeze(-1)

        # Attention mechanism
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date_map, y_date_map)
        A = torch.softmax(scale * scores, dim=-2)  # Softmax over L
        V = torch.einsum("bld,bdlp->bdp", x, A)

        y = self.norm1(self.dropout(V)).permute(0, 2, 1)

        # Fusion step using MLP for weighting
        error = x_enc_true - x_date_map.mean(-1)
        weight = self.MLP(error.permute(0, 2, 1)).unsqueeze(1)
        y_map = torch.stack([y_date_map.mean(-1), y], dim=-1)
        pred = torch.sum(y_map * weight, dim=-1)

        # Visualize attention if in 'test' mode
        if self.flag and mode == 'test':
            self.visual_attention(A)
            self.flag = False

        return pred


class Fusion_Block_s(nn.Module):
    def __init__(self, args, save_path):
        super(Fusion_Block_s, self).__init__()

        # Initialize layers and parameters
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)

        # Normalization layers
        self.graph_norm = nn.BatchNorm2d(args.d_feature)
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        self.norm3 = nn.BatchNorm1d(args.d_feature)

        # Output projection
        self.fc_out = nn.Linear(args.d_model, args.d_feature, bias=True)
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out = nn.Linear(patch_num * args.patch_len, args.pred_len)

        self.save_path = save_path
        self.flag = True

    def visual_attention(self, A):
        """Visualize attention heatmap"""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tmp = A[0, 0, :, :].clone().detach().cpu().numpy()
        plt.figure()
        ax = sns.heatmap(tmp, annot=False, fmt=".2e", cmap='coolwarm')
        cbar = ax.collections[0].colorbar
        decimal_places = 3
        ticks = np.linspace(tmp.min(), tmp.max(), num=5)
        tick_labels = [f'{tick:.{decimal_places}f}' for tick in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Attention Score')
        plt.savefig(os.path.join(self.save_path, '_Attention_Score'))
        return

    def forward(self, x, x_date, y_date, mode):
        """
        Forward pass of the Fusion Block s.

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, P_N, P_L)
            x_date (torch.Tensor): Date features for input tensor of shape (B, L, D, K)
            y_date (torch.Tensor): Date features for output tensor of shape (B, O, D, K)
            mode (str): Mode of operation ('train' or 'test')

        Returns:
            y (torch.Tensor): The final output tensor
        """
        B, L, D, k = x_date.shape
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.tanh(self.graph_norm(scores))

        # Calculate the output using attention mechanism
        V = torch.einsum("bld,bdlp->bdp", x, A)
        y = self.norm1(V)

        # Apply convolution layers and normalization
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = self.norm2((x + y)).transpose(-1, -2)

        # Visualize attention if in 'test' mode
        if self.flag and mode == 'test':
            self.visual_attention(A)
            self.flag = False

        return y
