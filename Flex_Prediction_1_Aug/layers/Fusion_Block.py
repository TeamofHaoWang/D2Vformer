import math
import torch
import torch.nn as nn
import os
import seaborn as sns
from layers.Date2Vec import *

class Fusion_Block_GLAFF(nn.Module):
    def __init__(self,args,save_path):
        super(Fusion_Block_GLAFF, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.activation=nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out=  nn.Linear(args.d_model, args.d_feature, bias=True)
        self.save_path = save_path
        self.flag = True
        self.q = 0.75

        self.MLP = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model),
            nn.GELU(),
            nn.Linear(args.d_model, 2),
            nn.Softmax(dim=-1)
        )


    def visual_attention(self, A):
        '''A:after softmax scores,维度也是四维的,并且最后两维是时间步间的相似性'''
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tmp = A[0, 0, :, :].clone().detach().cpu().numpy()
        plt.figure()
        ax = sns.heatmap(tmp, annot=False, fmt=".2e", cmap='coolwarm')
        # 获取图例对象
        cbar = ax.collections[0].colorbar
        # 设置刻度尺标签的小数位数
        decimal_places = 3  # 设置小数位数
        # 计算刻度步长
        ticks = np.linspace(tmp.min(), tmp.max(), num=5)  # 设置刻度数量
        # 将刻度值格式化为指定小数位数的字符串
        tick_labels = [f'{tick:.{decimal_places}f}' for tick in ticks]
        # 设置图例刻度尺标签
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Atten_Score')
        # plt.legend()
        plt.savefig(os.path.join(self.save_path, '_Atten_Score'))
        #plt.show()
        return




    def forward(self, x, x_date, y_date,mode):
        # x:(B,D,P_N,P_L),
        # x_date:(B,L,D,K),
        # y_date:(B,O,D,K),
        # return:y(B,O,D)

        B,L,D,k = x_date.shape

        # GLAFF
        x_enc_true = x   #TODO 因为之前已经revin过了，因此这里不再norm

        # denormalize
        robust_means_true = torch.median(x_enc_true, dim=1, keepdim=True)[0]
        robust_means_map = torch.median(x_date, dim=1, keepdim=True)[0]
        robust_stdev_true = torch.quantile(x_enc_true, self.q, 1, True) - torch.quantile(
            x_enc_true, 1 - self.q, 1, True) + 1e-6
        robust_stdev_map = torch.quantile(x_date, self.q, 1, True) - torch.quantile(
            x_date, 1 - self.q, 1, True) + 1e-6
        x_date_map = (x_date - robust_means_map) / robust_stdev_map * robust_stdev_true.unsqueeze(-1) + robust_means_true.unsqueeze(-1)
        y_date_map = (y_date - robust_means_map) / robust_stdev_map * robust_stdev_true.unsqueeze(-1) + robust_means_true.unsqueeze(-1)

        # Attention
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date_map, y_date_map)
        A = torch.softmax(scale * scores, dim=-2) # For L × O Softmax on L
        V = torch.einsum("bld,bdlp->bdp", x, A)
        # Attention norm

        y = self.norm1(self.dropout(V)).permute(0,2,1)

        # GLAFF
        error = x_enc_true - x_date_map.mean(-1)
        weight = self.MLP(error.permute(0, 2, 1)).unsqueeze(1)
        y_map = torch.stack([y_date_map.mean(-1), y], dim=-1)
        pred = torch.sum(y_map * weight, dim=-1)


        if self.flag == True and mode == 'test':
            self.visual_attention(A)
            self.flag = False

        return pred

class Fusion_Block_s(nn.Module):
    def __init__(self,args,save_path):
        super(Fusion_Block_s, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.activation=nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.graph_norm = nn.BatchNorm2d(args.d_feature)
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out=  nn.Linear(args.d_model, args.d_feature, bias=True)
        # Calculated directly from the formula
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out=nn.Linear(patch_num*args.patch_len,args.pred_len)
        self.save_path = save_path
        self.flag = True



    def visual_attention(self, A):
        '''A:after softmax scores,维度也是四维的,并且最后两维是时间步间的相似性'''
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tmp = A[0, 0, :, :].clone().detach().cpu().numpy()
        plt.figure()
        ax = sns.heatmap(tmp, annot=False, fmt=".2e", cmap='coolwarm')
        # 获取图例对象
        cbar = ax.collections[0].colorbar
        # 设置刻度尺标签的小数位数
        decimal_places = 3  # 设置小数位数
        # 计算刻度步长
        ticks = np.linspace(tmp.min(), tmp.max(), num=5)  # 设置刻度数量
        # 将刻度值格式化为指定小数位数的字符串
        tick_labels = [f'{tick:.{decimal_places}f}' for tick in ticks]
        # 设置图例刻度尺标签
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Atten_Score')
        # plt.legend()
        plt.savefig(os.path.join(self.save_path, '_Atten_Score'))
        #plt.show()
        return

    def forward(self, x, x_date, y_date,mode):

        B,L,D,k = x_date.shape
        # Attention
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.tanh(self.graph_norm(scores))  # 对于L×O 对L进行softmax

        V = torch.einsum("bld,bdlp->bdp", x, A)
        # Attention norm
        B,d,p=V.shape
        y = x = self.norm1(V)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        # ResNet+Attention norm
        y = self.norm2((x + y)).transpose(-1,-2)


        if self.flag == True and mode == 'test':
            self.visual_attention(A)
            self.flag = False

        return y



class Fusion_Block(nn.Module):
    def __init__(self,args,save_path):
        super(Fusion_Block, self).__init__()

        self.dropout = nn.Dropout(args.dropout)
        self.activation=nn.GELU()
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=args.d_feature, out_channels=args.d_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_feature, kernel_size=1)
        # Attention norm
        self.norm1 = nn.BatchNorm1d(args.d_feature)
        self.norm2 = nn.BatchNorm1d(args.d_feature)
        # Decoder norm
        self.norm3 = torch.nn.BatchNorm1d(args.d_feature)
        # Decoder projection
        self.fc_out=  nn.Linear(args.d_model, args.d_feature, bias=True)
        # Calculated directly from the formula
        patch_num = int((args.pred_len - args.patch_len) / args.stride + 1) + 1
        self.linear_out=nn.Linear(patch_num*args.patch_len,args.pred_len)
        self.save_path = save_path
        self.flag = True



    def visual_attention(self, A):
        '''A:after softmax scores,维度也是四维的,并且最后两维是时间步间的相似性'''
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tmp = A[0, 0, :, :].clone().detach().cpu().numpy()
        plt.figure()
        ax = sns.heatmap(tmp, annot=False, fmt=".2e", cmap='coolwarm')
        # 获取图例对象
        cbar = ax.collections[0].colorbar
        # 设置刻度尺标签的小数位数
        decimal_places = 3  # 设置小数位数
        # 计算刻度步长
        ticks = np.linspace(tmp.min(), tmp.max(), num=5)  # 设置刻度数量
        # 将刻度值格式化为指定小数位数的字符串
        tick_labels = [f'{tick:.{decimal_places}f}' for tick in ticks]
        # 设置图例刻度尺标签
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Atten_Score')
        # plt.legend()
        plt.savefig(os.path.join(self.save_path, '_Atten_Score'))
        #plt.show()
        return

    def forward(self, x, x_date, y_date,mode):

        B,L,D,k = x_date.shape
        # Attention
        scale = 1. / math.sqrt(k)
        scores = torch.einsum("bldk,bodk->bdlo", x_date, y_date)
        A = torch.softmax(scale * scores, dim=-2)  # 对于L×O 对L进行softmax
        V = torch.einsum("bdln,bdnp->bdlp", x.unsqueeze(2), A).squeeze()
        # Attention norm
        B,D,gap_len=V.shape
        x = V.reshape(B,D,-1)
        y = x = self.norm1(x)
        # Feed forward
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        # ResNet+Attention norm
        y = self.norm2((x + y)).transpose(-1,-2)


        if self.flag == True and mode == 'test':
            self.visual_attention(A)
            self.flag = False

        return y


class Fusion_Block_Mixer(nn.Module):
    def __init__(self,args):
        super(Fusion_Block_Mixer, self).__init__()

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

    def forward(self, x, x_date, y_date):
        # x:(B,D,P_N,P_L),
        # x_date:(B,L,D,K),
        # y_date:(B,O,D,K),
        # return:y(B,O,D)
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
    def __init__(self,args):
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
        self.linear_out = nn.Linear(patch_num*args.patch_len,args.pred_len)


        # Date2Vec
        self.multi_d2v_layers = torch.nn.ModuleList(
            [
                Date2Vec(args.d_model//(self.configs.down_sampling_window ** i), args.T2V_outmodel, args.d_feature, d_mark=args.d_mark)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

        self.multi_fusion_block = torch.nn.ModuleList(
            [
                Fusion_Block_Mixer(args)
                for i in range(self.configs.down_sampling_layers)
            ]
        )

        self.test_vision_flag = True


    def __multi_scale_process_inputs(self, x_enc, x_mark_enc, y_mark):

        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc.permute(0,2,1)
        y_mark_ori = y_mark.permute(0,2,1)

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
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window,:]
                if x_mark_enc_mark_ori.shape[1] != x_enc_ori.shape[2]:
                    x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :x_enc_ori.shape[2], :]
                x_mark_sampling_list.append(x_mark_enc_mark_ori)

            y_mark_ori =  y_mark_ori[:,::self.configs.down_sampling_window,:]
            y_mark_sampling_list.append(y_mark_ori)

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list
        y_mark_enc = y_mark_sampling_list


        return x_enc, x_mark_enc, y_mark_enc


    def forward(self, x, x_date, y_date,mode):

        B,D,H = x.shape
        _,M,H = x_date.shape
        _,M,p_l = y_date.shape

        # 获得多尺度的历史输入序列 以及 对应的历史mark

        x_enc, x_mark_enc, y_mark_enc = self.__multi_scale_process_inputs(x, x_date,y_date)
        multi_D2V_x = []
        multi_D2V_y = []
        multi_pred = []

        for i in range(self.configs.down_sampling_layers):
            p_l = y_mark_enc[i].shape[1]
            d2v_input = torch.cat([x_mark_enc[i].permute(0,2,1), y_mark_enc[i].permute(0,2,1)], dim=-1)
            d2V_output = self.multi_d2v_layers[i](x_enc[i].permute(0,2,1),d2v_input)  # [B,D,L+O,k]
            multi_D2V_x.append(d2V_output[:,:,:-p_l,:])
            multi_D2V_y.append(d2V_output[:, :, -p_l:, :])

            # pred[B,p_l,d]
            pred = self.multi_fusion_block[i](x_enc[i].permute(0,2,1),d2V_output[:,:,:-p_l,:],d2V_output[:, :, -p_l:, :])
            multi_pred.append(pred)

        pred = torch.zeros_like(multi_pred[0])
        for i in range(0,len(multi_pred)):
            scale_pred = multi_pred[i]
            b,l,d = scale_pred.shape
            scale_large_pred = scale_pred.repeat(1,2**i,1)
            scale_large_pred_reshape = scale_large_pred.reshape(b,2**i,-1,d).permute(0,2,1,3).reshape(b,-1,d)
            pred += scale_large_pred_reshape


        return pred





