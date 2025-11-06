import torch
import torch.nn as nn
import math
from layers.Revin import *
import matplotlib.pyplot as plt
import os
import seaborn as sns


class TimeVarKP_AR(nn.Module):
    """
    Koopman Predictor with DMD (analytical solution of the Koopman operator).
    Utilizes local variations within individual sliding windows to predict the future of time-variant terms.
    """

    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 seg_len=48,
                 dynamic_dim=128,
                 encoder=None,
                 decoder=None,
                 multistep=False,
                 ):
        super(TimeVarKP_AR, self).__init__()

        # Initialize parameters for model
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)  # Number of segments for input
        self.step = math.ceil(self.pred_len / self.seg_len)  # Number of segments for output
        self.padding_len = self.seg_len * self.freq - self.input_len

        # Koopman prediction layer
        self.dynamics = KPLayer_AR()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, C)

        Returns:
            x_rec (torch.Tensor): Reconstructed input tensor.
            x_pred (torch.Tensor): Predicted output tensor.
        """
        B, L, C = x.shape

        # Padding input for proper segmentation
        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)

        # Chunk the input into segments
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)

        # Encode the segmented input
        res = self.encoder(res)

        # Predict using the dynamics model
        x_rec, x_pred = self.dynamics(res, self.step)

        # Reconstruct input and reshape
        x_rec = self.decoder(x_rec)
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]

        # Predict future output
        x_pred = self.decoder(x_pred)
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]

        return x_rec, x_pred


class KPLayer_AR(nn.Module):
    """
    A demonstration of finding one-step transition of a linear system by DMD iteratively.
    """

    def __init__(self):
        super(KPLayer_AR, self).__init__()

        # Koopman operator (initially set to None)
        self.K = None  # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        """
        Perform one-step forward using the Koopman operator.

        Args:
            z (torch.Tensor): Input tensor (B x input_len x E)
            return_rec (bool): Whether to return reconstructed sequence.
            return_K (bool): Whether to return the Koopman operator.

        Returns:
            z_pred (torch.Tensor): Predicted next step in the sequence.
            z_rec (torch.Tensor): Reconstructed sequence if return_rec is True.
        """
        B, input_len, E = z.shape
        assert input_len > 1, 'Snapshots number should be larger than 1.'

        # Separate input and output for solving the linear system
        x, y = z[:, :-1], z[:, 1:]

        # Solve for the Koopman operator
        self.K = torch.linalg.lstsq(x, y).solution

        # Handle NaN in the Koopman operator by replacing it with an identity matrix
        if torch.isnan(self.K).any():
            print('Encountered NaN in K, replacing K with identity matrix.')
            self.K = torch.eye(self.K.shape[1]).to(self.K.device).unsqueeze(0).repeat(B, 1, 1)

        # Predict the next step
        z_pred = torch.bmm(z[:, -1:, :], self.K)

        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred

    def forward(self, z, pred_len=1):
        """
        Forward pass for multi-step prediction.

        Args:
            z (torch.Tensor): Input tensor (B x input_len x E)
            pred_len (int): Number of steps to predict.

        Returns:
            z_rec (torch.Tensor): Reconstructed input tensor.
            z_preds (torch.Tensor): Predicted sequence tensor.
        """
        assert pred_len >= 1, 'Prediction length should not be less than 1.'

        # One-step prediction and reconstruction
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)

        # Predict subsequent steps iteratively
        z_preds = [z_pred]
        for i in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)

        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds
