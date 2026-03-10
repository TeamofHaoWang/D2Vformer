import torch
import torch.nn as nn
from layers.koopa_related import *
import math
from layers.RevIN import *
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineActivation_D2V(nn.Module):
    def __init__(self, in_features, out_features, d_features, d_mark, save_path):
        # 对于D2V而言，这里的输入in_features 指代为 seq_len ，out_features指代对应每一个D用多少个正弦函数来表征
        super(SineActivation_D2V, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(d_features, 1, 1, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        nn.init.uniform_(self.w, 0, 2 * torch.pi)
        self.b = nn.parameter.Parameter(torch.randn(d_features, 1, 1, out_features - 1))
        self.f = torch.sin
        self.save_path = save_path
        self.call_flag = False  # 判断有没有画过图的标志

        self.w_transform_t = nn.Parameter(torch.randn((d_mark, d_features)), requires_grad=True)  # 测试过使用不同的FC效果好一点
        self.w_transform_s = nn.Parameter(torch.randn((d_mark, d_features)), requires_grad=True)

    def forward(self, data, tau, flag=None):
        output = self.t2v(data, tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

        return output

    def t2v(self, data, tau, f, out_features, w, b, w0, b0, arg=None):
        _, D, _ = data.shape
        tau = tau.unsqueeze(-1).repeat(1, 1, 1, D)  # (B,d_mark,L+O,d_feature)
        tau = tau.transpose(1, -1)  # (B,d_feature,L+O,d_mark)
        mark = tau.shape[-1]
        w_trend = torch.matmul(data, w0).unsqueeze(-2)
        w_season = torch.matmul(data, w).unsqueeze(-2)

        # 相当于多出了一个d_mark维度，并且对待不同的mark都是采用一个w
        w_trend = w_trend.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        w_season = w_season.unsqueeze(2).repeat(1, 1, mark, 1, 1)
        tau = tau.unsqueeze(2).transpose(2, -1)
        v2 = torch.matmul(tau, w_trend) + b0
        v1 = f(torch.matmul(tau, w_season) + b)

        self.w_trend = w_trend
        self.w_season = w_season
        v1 = torch.mean(v1, dim=2)  # 将mark维度融掉 (B,D,L,k)
        v2 = torch.mean(v2, dim=2)  # 将mark维度融掉 (B,D,L,k)

        return torch.cat([v2, v1], -1)

    def visual(self, w, save_path, type):
        '''The dimensions of w are four-dimensional, and the last dimension represents how many frequency domain components are used to represent the seasonal term'''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 画训练后的权重图
        tmp = w.clone()
        # 图1
        plt.figure()
        result = tmp[0, 0, 0, :]
        remainder = torch.remainder(result, 2 * torch.tensor(np.pi))  # 计算tmp取余2π的余数
        remainder = torch.where(result < 0, remainder * (-1), remainder)
        # 输出限制后的tmp值
        plt.scatter(np.arange(0, len(tmp[0, 0, 0, :])), remainder.detach().cpu().numpy(), label='w')
        plt.title(type + '_w')
        plt.xlabel('num of w')
        plt.ylabel('values of w')
        plt.legend()
        plt.savefig(os.path.join(save_path, type + '_w'))
        plt.close()
        return


class Fusion_Block_Mixer(nn.Module):
    def __init__(self, args):
        super(Fusion_Block_Mixer, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature+1, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature+1, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature+1)
        self.norm2 = nn.BatchNorm1d(args.d_feature+1)



    def visual_attention(self, A, save_path):
        '''A:After softmax scores, the dimensions are also four-dimensional, and the last two dimensions are similarities between time steps'''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp = A[0, 0, :, :].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm')
        plt.title('Atten_Score')
        # plt.legend()
        plt.savefig(os.path.join(save_path, '_Atten_Score'))
        plt.close()
        return

    def forward(self, x, x_date, y_date, flag=None):
        x_date = x_date.permute(0, 2, 1, 3)
        y_date = y_date.permute(0, 2, 1, 3)
        B, L, D, k = x_date.shape
        # Attention
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.softmax(scale * scores, dim=-2)  # For L × O Softmax on L
        V = torch.einsum("bdn,bdnp->bdp", x, A)
        # Attention norm
        B, D, _, = V.shape
        x = V
        y = x = self.norm1(x)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        # ResNet+Attention norm
        y = self.norm2((x + y)).transpose(-1, -2)

        return y


class Multi_D2V(nn.Module):
    def __init__(self, args):
        super(Multi_D2V, self).__init__()

        self.configs = args
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out = nn.Linear(args.d_model, args.d_feature, bias=True)
        # Calculated directly from the formula
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out = nn.Linear(patch_num * args.patch_len, args.pred_len)

        # Date2Vec
        self.multi_d2v_layers = torch.nn.ModuleList(
            [
                SineActivation_D2V(args.d_model // (self.configs.down_sampling_window ** i), args.T2V_outmodel,
                                   args.d_feature, d_mark=args.d_mark, save_path=args.output_path)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

        self.multi_fusion_block = torch.nn.ModuleList(
            [
                Fusion_Block_Mixer(args)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc, y_mark, flag=None):

        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc.permute(0, 2, 1)
        y_mark_ori = y_mark.permute(0, 2, 1)

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        y_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc_mark_ori)
        y_mark_sampling_list.append(y_mark_ori)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]
                if x_mark_enc_mark_ori.shape[1] != x_enc_ori.shape[2]:
                    x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :x_enc_ori.shape[2], :]
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

            y_mark_ori = y_mark_ori[:, ::self.configs.down_sampling_window, :]
            y_mark_sampling_list.append(y_mark_ori)

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list
        y_mark_enc = y_mark_sampling_list

        return x_enc, x_mark_enc, y_mark_enc

    def forward(self, x, x_date, y_date, flag=None):

        B, D, H = x.shape
        _, M, H = x_date.shape
        _, M, p_l = y_date.shape

        # Obtaining multi-scale historical input sequences and corresponding historical marks

        x_enc, x_mark_enc, y_mark_enc = self.__multi_scale_process_inputs(x, x_date, y_date)
        multi_D2V_x = []
        multi_D2V_y = []
        multi_pred = []

        for i in range(self.configs.down_sampling_layers):
            p_l = y_mark_enc[i].shape[1]
            d2v_input = torch.cat([x_mark_enc[i].permute(0, 2, 1), y_mark_enc[i].permute(0, 2, 1)], dim=-1)
            d2V_output = self.multi_d2v_layers[i](x_enc[i].permute(0, 2, 1), d2v_input)  # [B,D,L+O,k]
            multi_D2V_x.append(d2V_output[:, :, :-p_l, :])
            multi_D2V_y.append(d2V_output[:, :, -p_l:, :])

            # pred[B,p_l,d]
            pred = self.multi_fusion_block[i](x_enc[i].permute(0, 2, 1), d2V_output[:, :, :-p_l, :],
                                              d2V_output[:, :, -p_l:, :])
            multi_pred.append(pred)

        pred = torch.zeros_like(multi_pred[0])
        for i in range(0, len(multi_pred)):
            scale_pred = multi_pred[i]
            b, l, d = scale_pred.shape
            scale_large_pred = scale_pred.repeat(1, 2 ** i, 1)
            scale_large_pred_reshape = scale_large_pred.reshape(b, 2 ** i, -1, d).permute(0, 2, 1, 3).reshape(b, -1, d)
            # print('scale_large_pred_reshape:',scale_large_pred_reshape.shape)
            # print('pred:',pred.shape)
            # print('multi_pred:',multi_pred[i].shape)
            pred += scale_large_pred_reshape

        return pred



class Multi_D2V_non_blind(nn.Module):
    def __init__(self, args):
        super(Multi_D2V_non_blind, self).__init__()

        self.configs = args
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out = nn.Linear(args.d_model, args.d_feature, bias=True)
        # Calculated directly from the formula
        # patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        # self.linear_out = nn.Linear(patch_num * args.patch_len, args.pred_len)

        # Date2Vec
        self.multi_d2v_layers = torch.nn.ModuleList(
            [
                SineActivation_D2V(args.d_model // (self.configs.down_sampling_window ** i), args.T2V_outmodel,
                                   args.d_feature+1, d_mark=args.d_mark, save_path=args.output_path)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

        self.multi_fusion_block = torch.nn.ModuleList(
            [
                Fusion_Block_Mixer(args)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc, y_mark, flag=None):

        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc.permute(0, 2, 1)
        y_mark_ori = y_mark.permute(0, 2, 1)

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        y_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc_mark_ori)
        y_mark_sampling_list.append(y_mark_ori)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]
                if x_mark_enc_mark_ori.shape[1] != x_enc_ori.shape[2]:
                    x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :x_enc_ori.shape[2], :]
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

            y_mark_ori = y_mark_ori[:, ::self.configs.down_sampling_window, :]
            y_mark_sampling_list.append(y_mark_ori)

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list
        y_mark_enc = y_mark_sampling_list

        return x_enc, x_mark_enc, y_mark_enc

    def forward(self, x, x_date, y_date, flag=None):

        B, D, H = x.shape
        _, M, H = x_date.shape
        _, M, p_l = y_date.shape

        # 获得多尺度的历史输入序列 以及 对应的历史mark

        x_enc, x_mark_enc, y_mark_enc = self.__multi_scale_process_inputs(x, x_date, y_date)
        multi_D2V_x = []
        multi_D2V_y = []
        multi_pred = []

        for i in range(self.configs.down_sampling_layers):
            p_l = y_mark_enc[i].shape[1]
            d2v_input = torch.cat([x_mark_enc[i].permute(0, 2, 1), y_mark_enc[i].permute(0, 2, 1)], dim=-1)
            d2V_output = self.multi_d2v_layers[i](x_enc[i].permute(0, 2, 1), d2v_input)  # [B,D,L+O,k]
            multi_D2V_x.append(d2V_output[:, :, :-p_l, :])
            multi_D2V_y.append(d2V_output[:, :, -p_l:, :])

            # pred[B,p_l,d]
            pred = self.multi_fusion_block[i](x_enc[i].permute(0, 2, 1), d2V_output[:, :, :-p_l, :],
                                              d2V_output[:, :, -p_l:, :])
            multi_pred.append(pred)

        pred = torch.zeros_like(multi_pred[0])
        for i in range(0, len(multi_pred)):
            scale_pred = multi_pred[i]
            b, l, d = scale_pred.shape
            scale_large_pred = scale_pred.repeat(1, 2 ** i, 1)
            scale_large_pred_reshape = scale_large_pred.reshape(b, 2 ** i, -1, d).permute(0, 2, 1, 3).reshape(b, -1, d)
            # print('scale_large_pred_reshape:',scale_large_pred_reshape.shape)
            # print('pred:',pred.shape)
            # print('multi_pred:',multi_pred[i].shape)
            pred += scale_large_pred_reshape

        return pred


class RangeMLP(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # x.shape: (B, 2) → output.shape: (B, out_dim)

class D2Vformer_TSI(nn.Module):
    def __init__(self, configs, train_gap_len):
        super(D2Vformer_TSI, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model
        self.train_gap_len = train_gap_len

        # non blind
        self.predlen_MLP = RangeMLP(self.seq_len)
        self.d_non_blind = self.d_feature + 1

        # remove non blind
        self.fc_out = nn.Conv1d(in_channels=self.d_non_blind, out_channels=self.d_feature, kernel_size=1)

        # Encoder
        self.all_linear = nn.Linear(self.seq_len, self.d_model)
        self.trend_linear = nn.Linear(self.seq_len, self.d_model)
        self.season_linear = nn.Linear(self.seq_len, self.d_model)

        # self.trend_linear_decoder = nn.Linear(self.d_model, self.pred_len)

        # Mapping time-invariant data
        K_init = torch.randn(self.d_model, self.d_model)
        U, _, V = torch.svd(K_init)  # stable initialization
        self.K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

        # Decoder
        # self.project = nn.Linear(self.d_model, self.pred_len)

        # 对时间信息进行编码
        self.linear_t = nn.Linear(self.seq_len, self.d_model)
        self.linear_t_out = nn.Linear(self.d_model, self.seq_len)

        # Revin
        self.revin = RevIN(self.d_feature, affine=True, subtract_last=False)

        # Time2Vec
        self.Time2Vec = SineActivation_D2V(self.d_model, configs.T2V_outmodel, self.d_feature, d_mark=configs.d_mark,
                                           save_path=configs.output_path)

        # fusion
        self.multi_d2v = Multi_D2V_non_blind(args=configs)
        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # Fourier splitting-->Decomposition of timing data
        self.mask_spectrum = configs.mask_spectrum
        self.disentanglement = FourierFilter(self.mask_spectrum)

        # 可视化
        self.save_path = configs.output_path
        self.call_flag = False
        # BN层
        self.bn_inv = nn.BatchNorm1d(self.d_model)
        self.bn_var = nn.BatchNorm1d(self.d_non_blind)
        self.bn_pred = nn.BatchNorm1d(self.d_feature)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, flag=None):
        # 由于跳跃预测的要求，模型的建构中不应当出现predlen这种参数来调节预测长度，因此对于原模型作出一定程度的修改
        # x_mark_dec经过修改应当只包含预测位置的日期，不存在label_len这种东西。
        # 也就意味着所有通过Linear输出预测结果的方式不存在了，只能通过Fusion的方式输出，因为Fusion不需要预定义输出长度。

        B, L, D = x_enc.shape
        _, gap_len, _ = x_dec.shape
        slice_len = x_mark_dec.shape[1]
        x_enc = self.revin(x_enc, 'norm')
        # non blind
        range_tensor = torch.tensor([gap_len / self.train_gap_len[0] , gap_len / self.train_gap_len[-1]])
        pred_len_feature = self.predlen_MLP(range_tensor.float().unsqueeze(0).to(x_enc.device).expand(B, 2))
        x_in = torch.cat([x_enc,pred_len_feature.unsqueeze(-1)],dim=-1)


        # Decomposition into time-varying and time-invariant components
        x_var, x_inv = self.disentanglement(x_in)
        '''The following treatment of time-invariant components'''
        # The duration L of the time-invariant information is mapped to the high-dimensional d_model-->Encoder
        trend_output = self.all_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        season_output = self.season_linear(x_inv.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_output = trend_output + season_output  # (B,L,D)
        encoder_output = self.bn_inv(encoder_output)
        y_inv = self.K(encoder_output.transpose(-1, -2)).transpose(-1, -2)  # Koopa中时不变中的全局K算子
        # Decoder
        y_inv = y_inv[:, -slice_len:, :]  # (B,O,D)
        # print('y_inv:',y_inv.shape)

        '''The following operations are performed on time-varying data'''
        # Getting the date input for D2V
        D2V_input = torch.cat((x_mark_enc, x_mark_dec[:, -slice_len:, :]), dim=-2)
        D2V_input = D2V_input.transpose(-1, -2)  # (B,D,L)

        # Encoding the time code of history
        t_history = self.linear_t(D2V_input[:, :, :self.seq_len])
        D2V_input = torch.cat([t_history, D2V_input[:, :, -slice_len:]], dim=-1)  # (B,d_mark,L+O)

        all_output_var = self.all_linear(x_var.permute(0, 2, 1))
        season_output_var = self.season_linear(x_var.permute(0, 2, 1))
        trend_output_var = (all_output_var - season_output_var).permute(0, 2, 1)
        trend_output_var = trend_output_var[:, -slice_len:, :]
        # print('trend_output_var:',trend_output_var.shape)

        encoder_output_var = season_output_var
        encoder_output_var = self.bn_var(encoder_output_var)

        dec_out = self.multi_d2v(encoder_output_var, t_history, D2V_input[:, :, -slice_len:])
        # print('dec_out:',dec_out.shape)
        prediction = dec_out + y_inv + trend_output_var

        # Denorm
        prediction = self.revin(self.fc_out(prediction.transpose(-2,-1)).transpose(-2,-1), 'denorm')

        return prediction  # [B, O, D]




