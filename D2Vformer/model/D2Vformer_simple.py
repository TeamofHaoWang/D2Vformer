from layers.Date2Vec import *
from layers.Fusion_Block import *
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from layers.Revin import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Flexible Prediction version for D2Vformer

'''

class D2Vformer_simple(nn.Module):
    def __init__(self, configs):
        super(D2Vformer_simple, self).__init__()
        # Changing variable names
        self.configs=configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.d_feature = configs.d_feature
        self.d_model = configs.d_model
        self.mark_index = configs.mark_index

        #Revin
        self.revin = RevIN(self.d_feature, affine=False, subtract_last= False)

        # Date2Vec
        self.Date2Vec = Date2Vec_Fourier(self.seq_len,configs.T2V_outmodel,self.d_feature,d_mark=len(self.mark_index),save_path = configs.output_path)

        # Fusion
        self.fusion = Fusion_Block_s(args=configs,save_path=configs.output_path)
        self.patch_len=configs.patch_len
        self.stride=configs.stride

        # BN Layer
        self.bn_inv=nn.BatchNorm1d(self.d_model)
        self.bn_var = nn.BatchNorm1d(self.d_feature)
        self.bn_pred=nn.BatchNorm1d(self.d_feature)

    def do_patching(self,z,stride):
        # Patching
        z=nn.functional.pad(z, (0, stride))
        # real doing patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        return z


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,mode):
        # revin   in[B,L,D]    out[B,L,D]
        x_enc = self.revin(x_enc, 'norm')

        '''The following operations are performed on time-varying data'''
        # Getting the date input for D2V
        D2V_input=torch.cat((x_mark_enc,x_mark_dec[:,-self.pred_len:,:]),dim=-2)
        D2V_input=D2V_input[:, :, self.mark_index].transpose(-1,-2) #(B,D,L)

        # The relationship between historical and predicted positions is extracted by inputting the encoded historical information into D2V
        D2V_output = self.Date2Vec(x_enc.permute(0,2,1), D2V_input, mode)  # [B,D,L+O,k]
        D2V_output = D2V_output.permute(0, 2, 1, 3)

        D2V_x_date = D2V_output[:, :self.seq_len, :, :]
        # Date coding of model predicted locations
        D2V_y_date = D2V_output[:, -self.pred_len:, :, :]

        # Based on the extracted future and history DATE relationships, and the input history sequence, future results are obtained
        prediction_var = self.fusion(x_enc, D2V_x_date, D2V_y_date, mode)

        # Integrate the output of each component
        prediction=prediction_var
        prediction=self.bn_pred(prediction.transpose(-1,-2)).transpose(-1,-2)

        # Denorm
        prediction = self.revin(prediction, 'denorm')

        return prediction # [B, O, D]






