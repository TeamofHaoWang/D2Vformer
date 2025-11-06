# -*- coding: utf-8 -*-
import os
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from exp.exp import EXP
from utils.setseed import set_seed

if __name__ == '__main__':

    # Convert string to boolean for argparse argument parsing
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v == 'True'

    import argparse
    parser = argparse.ArgumentParser(description="Model configuration and training settings")

    # Experiment model parameters
    parser.add_argument('--model_name', default='PatchTST', type=str,
                        help='Model options: [T2V_Transformer, T2V_ITransformer, T2V_PatchTST, D2V_Fourier_Transformer, '
                             'D2V_Fourier_PatchTST, D2V_Fourier_ITransformer, GLAFF_ITransformer, GLAFF_PatchTST, '
                             'GLAFF_Transformer, Transformer, PatchTST, ITransformer]')
    parser.add_argument('--train', default=True, type=str2bool, help='Whether to train the model')
    parser.add_argument('--resume', default=False, type=str2bool, help='Resume from checkpoint')
    parser.add_argument('--save_log', default=True, type=str2bool, help='Save training logs')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (None for auto +1)')
    parser.add_argument('--resume_dir', type=str, default=None, help='Directory to resume from checkpoint')
    parser.add_argument('--loss', default='normal', type=str, help='Loss function options: [quantile, normal]')
    parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')

    # Experiment type and description
    parser.add_argument('--exp', default='deep_learning', type=str, help='Experiment type [deep_learning]')
    parser.add_argument('--exp_num', type=int, default=-1)
    parser.add_argument('--desc', type=str, default='', help='Description of the current experiment')

    # Data-related parameters
    parser.add_argument('--data_name', default='traffic', type=str,
                        help='Dataset options: [ETTh1, electricity, exchange, ETTm1, traffic]')
    parser.add_argument('--seq_len', default=96, type=int, help='Input sequence length')
    parser.add_argument('--label_len', default=48, type=int, help='Decoder input length for transformer')
    parser.add_argument('--pred_len', default=96, type=int, help='Prediction length')
    parser.add_argument('--d_mark', default=27, type=int, help='Date embedding dimension')
    parser.add_argument('--d_feature', default=862, type=int,
                        help='Input data feature dimension without date: [ETTh1: 7, electricity: 321, exchange: 8, '
                             'ETTm1: 7, illness: 7, traffic: 862]')
    parser.add_argument('--c_out', type=int, default=862, help='Output size')

    # Forecasting task options
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task options: [M, S, MS]; M: multivariate to multivariate, S: univariate to univariate, MS: multivariate to univariate')

    # Hyperparameters
    parser.add_argument('--d_model', default=510, type=int, help='Feature dimension in model (must be divisible by 3 for Fedformer)')
    parser.add_argument('--d_ff', default=1024, type=int, help='Feature dimension 2 in model')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--lr', default=0.000001, type=float, help='Initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='Learning rate for GAN discriminator')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--epoches', default=150, type=int, help='Number of training epochs')
    parser.add_argument('--patience', default=5, type=int, help='Early stopping patience')

    # Quantile loss parameters
    parser.add_argument('--quantiles', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=int, help='Quantile loss ρ parameters')

    # Model architecture parameters
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of attention heads (must be 3 for Fedformer)')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')

    # Patch parameters
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length')
    parser.add_argument('--stride', type=int, default=8, help='Stride between patches')
    parser.add_argument('--patch_stride', type=int, default=8)

    # Fourier decomposition parameters
    parser.add_argument('--fourier_decomp_ratio', type=float, default=0.5, help='Fourier decomposition ratio')
    parser.add_argument('--stronger', type=int, default=2, help='Top-K Fourier bases')

    # D2V parameters
    parser.add_argument('--T2V_outmodel', type=int, default=64, help='T2V output model dimension')

    # ITransformer parameters
    parser.add_argument('--embed', type=str, default='timeF', help='Time feature encoding: [timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features encoding: [s: secondly, t: minutely, h: hourly, d: daily, '
                             'b: business days, w: weekly, m: monthly]')

    # Autoformer parameters
    parser.add_argument('--moving_avg', default=[24], help='Window size for moving average')

    # Fedformer parameters
    parser.add_argument('--mode_select', type=str, default='random', help='Fedformer mode selection: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='Number of modes to be selected randomly')

    # FEB Denoising
    parser.add_argument('--weight_decay', default=0.001, type=float)

    # GLAFF plugin parameters
    parser.add_argument('--dim', type=int, default=256, help='Dimension of hidden state')
    parser.add_argument('--dff', type=int, default=512, help='Dimension of feed-forward network')
    parser.add_argument('--head_num', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout2', type=float, default=0.6, help='Dropout rate of plugin')
    parser.add_argument('--q', type=float, default=0.75, help='Quantile for loss function')

    # Additional info
    parser.add_argument('--info', type=str, default='None', help='Extra information')

    # Parse arguments
    args = parser.parse_args()
    set_seed(args.seed)

    # Print arguments for verification
    print(f"|{'=' * 101}|")
    for key, value in args.__dict__.items():
        print(f"|{str(key):>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")

    # Initialize and run experiment
    exp = EXP(args)
    if args.train:
        exp.train()  # Train the model
    exp.test()  # Test the model
