# -*- coding: utf-8 -*-
import os
import io
import sys
import argparse
from exp.exp import EXP
from utils.setseed import set_seed

# Set environment for CUDA and UTF-8 encoding
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Convert string to boolean for argparse argument parsing
def str2bool(v):
    return v.lower() in ('true', '1')

if __name__ == '__main__':
    # Argument parser for model parameters and configurations
    parser = argparse.ArgumentParser(description="Model configuration and training settings")

    # Model related arguments
    parser.add_argument('--model_name', default='LLM', type=str, help='Model name options: [D2Vformer, D2Vformer_s.]')
    parser.add_argument('--train', default=True, type=str2bool, help='Whether to train the model')
    parser.add_argument('--resume', default=False, type=str2bool, help='Resume training from checkpoint')
    parser.add_argument('--save_log', default=True, type=str2bool, help='Whether to save log')
    parser.add_argument('--output_dir', type=str, help='Directory for output results')
    parser.add_argument('--resume_dir', type=str, help='Directory for checkpoint to resume from')
    parser.add_argument('--loss', default='normal', type=str, help='Loss function options: [quantile, normal]')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    # Experiment specific parameters
    parser.add_argument('--exp', default='deep_learning', type=str, help='Experiment type')
    parser.add_argument('--exp_num', type=int, default=-1, help='Experiment number')
    parser.add_argument('--desc', type=str, help='Description of the experiment')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay')

    # Data related parameters
    parser.add_argument('--data_name', default='ETTh1', type=str, help='Data name options: [ETTh1, electricity, etc.]')
    parser.add_argument('--seq_len', default=96, type=int, help='Input sequence length')
    parser.add_argument('--label_len', default=48, type=int, help='Decoder input length')
    parser.add_argument('--pred_len', default=96, type=int, help='Prediction length')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--d_mark', default=27, type=int, help='Date embedding dimension')
    parser.add_argument('--d_feature', default=7, type=int, help='Feature dimension without date')
    parser.add_argument('--c_out', type=int, default=7, help='Output size')
    parser.add_argument('--features', type=str, default='M', help='Feature forecasting options: [M, S, MS]')

    # Hyperparameters
    parser.add_argument('--d_model', default=512, type=int, help='Model feature dimension')
    parser.add_argument('--d_ff', default=1024, type=int, help='Feedforward feature dimension')
    parser.add_argument('--dropout', default=0.05, type=float, help='Dropout rate')
    parser.add_argument('--lr', default=0.0005, type=float, help='Initial learning rate')
    parser.add_argument('--lr_d', default=0.05, type=float, help='Learning rate for GAN discriminator')

    parser.add_argument('--epoches', default=200, type=int, help='Number of epochs for training')
    parser.add_argument('--patience', default=5, type=int, help='Patience for early stopping')

    # Quantile loss parameters
    parser.add_argument('--quantiles', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], type=int, help='Quantile loss parameters')

    # Additional experiment information
    parser.add_argument('--info', type=str, default='None', help='Additional info for experiment')

    # Patch related parameters
    parser.add_argument('--patch_len', default=16, type=int, help='Patch length')
    parser.add_argument('--stride', default=8, type=int, help='Stride for patches')

    # T2V model specific parameters
    parser.add_argument('--T2V_outmodel', default=64, type=int, help='T2V output model size')
    parser.add_argument('--d2v_in_feature', default=7, type=int, help='D2V input feature size')

    # Time feature encoding
    parser.add_argument('--embed', default='timeF', type=str, help='Time feature encoding method')
    parser.add_argument('--freq', default='h', type=str, help='Frequency for time encoding')

    # Layer and head configurations
    parser.add_argument('--e_layers', default=2, type=int, help='Number of encoder layers')
    parser.add_argument('--d_layers', default=1, type=int, help='Number of decoder layers')
    parser.add_argument('--n_heads', default=3, type=int, help='Number of attention heads')
    parser.add_argument('--activation', default='gelu', type=str, help='Activation function')

    # TimeMixer specific parameters
    parser.add_argument('--channel_independence_timemixer', default=1, type=int, help='Channel independence for TimeMixer')
    parser.add_argument('--decomp_method', default='moving_avg', type=str, help='Decomposition method for TimeMixer')
    parser.add_argument('--use_norm_timemixer', default=1, type=int, help='Whether to use normalization in TimeMixer')
    parser.add_argument('--down_sampling_layers', default=3, type=int, help='Number of down sampling layers')
    parser.add_argument('--down_sampling_window', default=2, type=int, help='Window size for down sampling')

    # LLM specific parameters
    parser.add_argument('--token_len', default=16, type=int, help='Token length for LLM')
    parser.add_argument('--llm_ckp_dir', default=r'E:\other_reproduce_models\time_series\AutoTimes-main\AutoTimes-main\GPT2', type=str, help='LLM checkpoint directory')

    # MLP layers for LLM
    parser.add_argument('--mlp_hidden_dim', default=256, type=int, help='MLP hidden dimension')
    parser.add_argument('--mlp_hidden_layers', default=2, type=int, help='Number of hidden layers in MLP')
    parser.add_argument('--mlp_activation', default='tanh', type=str, help='Activation function for MLP')

    # Parse the arguments
    args = parser.parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)

    # Print the arguments
    print(f"|{'=' * 101}|")
    for key, value in args.__dict__.items():
        print(f"|{key:>50s}|{str(value):<50s}|")
    print(f"|{'=' * 101}|")

    # Execute the experiment
    if args.exp == 'deep_learning':
        exp = EXP(args)
        if args.train:
            exp.train()  # Train the model
        exp.test()  # Test the model
