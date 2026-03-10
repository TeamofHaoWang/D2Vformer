import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.d_feature
        self.Real = configs.Real
        self.train_mode = configs.train_mode
        self.configs = configs
        self.reconstruct = configs.reconstruct

        self.dominance_freq = configs.cut_freq  # 720/24
        if self.dominance_freq == 0:
            self.dominance_freq = int(self.seq_len // configs.base_T + 1) * configs.H_order + 10

        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # if Weight Sharing, individual is False
        if not self.Real:
            if self.individual:
                self.freq_upsampler = nn.ModuleList()
                for i in range(self.channels):
                    self.freq_upsampler.append(
                        nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))

            else:
                self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                    torch.cfloat)  # complex layer for frequency upcampling]

        else:
            # This is the real value implementation of the original FITS.
            # Real_FITS simulates the complex value multiplication with two layer of real value linear layer following
            # Y_real = X_real*W_real - X_imag * W_imag
            # Y_imag = X_real*W_imag + X_imag * W_real
            if self.individual:
                self.freq_upsampler_real = nn.ModuleList()
                self.freq_upsampler_imag = nn.ModuleList()
                for i in range(self.channels):
                    self.freq_upsampler_real.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
                    self.freq_upsampler_imag.append(nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))

            else:
                self.freq_upsampler_real = nn.Linear(self.dominance_freq,
                                                     int(self.dominance_freq * self.length_ratio))  # complex layer for frequency upcampling]
                self.freq_upsampler_imag = nn.Linear(self.dominance_freq,
                                                     int(self.dominance_freq * self.length_ratio))  # complex layer for frequency upcampling]

    def loss_fn(self, outputs, x, y):
        # FITS have 3 training mode:
        # 0:train on y   1: train on xy  2: train on xy and finetune on y
        # if you want to train on mode 2, you should train on mode 1 first then on mode 0
        criterion = nn.MSELoss()
        f_dim = -1 if self.configs.features == 'MS' else 0
        y = y[:, -self.configs.pred_len:, f_dim:].to(x.device)
        if self.train_mode == 0:
            loss = criterion(outputs, y)
        elif self.train_mode == 1:
            xy = torch.concat([x, y], dim=1)
            loss = criterion(outputs, xy)
        else:
            raise 'if you want to train on mode 2, you should train on mode 1 first then on mode 0'

        return loss


    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark,mode):
        # RIN
        f_dim = -1 if self.configs.features == 'MS' else 0

        if self.reconstruct:
            batch_x = batch_y[:, -self.pred_len:, f_dim:].to(batch_x.device)

        if self.train_mode == 0:
            x = batch_x
        elif self.train_mode == 1:
            x = torch.concat([batch_x, batch_y[:, -self.pred_len:, f_dim:].to(batch_x.device)], dim=1)
        else:
            raise 'if you want to train on mode 2, you should train on mode 1 first then on mode 0'

        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        if low_specx.shape[1] < self.dominance_freq:
            low_specx = torch.fft.rfft(x, n=2*(self.dominance_freq-1), dim=1)

        if not self.Real:
            low_specx[:, self.dominance_freq:] = 0  # LPF
            low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF

            if self.individual:
                low_specxy_ = torch.zeros(
                    [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                    dtype=low_specx.dtype).to(low_specx.device)
                for i in range(self.channels):
                    low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
            else:
                low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

            low_specxy = torch.zeros(
                [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
                dtype=low_specxy_.dtype).to(low_specxy_.device)
            low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding


        else:
            low_specx = torch.view_as_real(low_specx[:, 0:self.dominance_freq, :])
            low_specx_real = low_specx[:, :, :, 0]
            low_specx_imag = low_specx[:, :, :, 1]
            if self.individual:
                # The following content was not used
                low_specxy_ = torch.zeros([low_specx.size(0), int(self.cut_freq * self.length_ratio), low_specx.size(2)],
                                          dtype=low_specx.dtype).to(low_specx.device)
                for i in range(self.channels):
                    low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
            else:
                real = self.freq_upsampler_real(low_specx_real.permute(0, 2, 1)).permute(0, 2, 1)
                imag = self.freq_upsampler_imag(low_specx_imag.permute(0, 2, 1)).permute(0, 2, 1)
                low_specxy_real = real - imag
                low_specxy_imag = real + imag

            # zero padding
            low_specxy_R = torch.zeros(
                [low_specxy_real.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_real.size(2)],
                dtype=low_specxy_real.dtype).to(low_specxy_real.device)
            low_specxy_R[:, 0:low_specxy_real.size(1), :] = low_specxy_real

            low_specxy_I = torch.zeros([low_specxy_imag.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_imag.size(2)],
                                       dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)
            low_specxy_I[:, 0:low_specxy_imag.size(1), :] = low_specxy_imag

            low_specxy = torch.complex(low_specxy_R, low_specxy_I)

        n = self.pred_len if self.train_mode == 0 else self.pred_len + self.seq_len

        low_xy = torch.fft.irfft(low_specxy, n=n, dim=1)

        low_xy = low_xy * self.length_ratio  # compemsate the length change

        # irRIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean

        #loss = self.loss_fn(xy, batch_x, batch_y)

        return xy