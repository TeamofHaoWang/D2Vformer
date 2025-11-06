import os
import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
from layers.Quantile_loss import *
import utils.config
import utils.save
from datetime import datetime


class EXP:
    def __init__(self, args):
        assert args.resume_dir == args.output_dir
        if args.output_dir is None or args.output_dir == 'None' or args.output_dir == 'none':
            args.output_dir = None
            utils.config.create_output_dir(args)  # Create output directory
            args.resume_dir = args.output_dir
        else:
            args.output_dir = os.path.join('experiments', args.output_dir)
            args.resume_dir = os.path.join('experiments', args.resume_dir)
        output_path = os.path.join(args.output_dir, args.model_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.output_path = output_path
        resume_path = os.path.join(args.resume_dir, args.model_name)
        if not os.path.exists(resume_path):
            raise ValueError('Checkpoint path for pretrained weights not found')
        resume_path = os.path.join(resume_path, args.data_name + '_best_model.pkl')
        self.resume_path = resume_path

        # Initialize other parameters
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len
        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        self.args = args

        # Setup GPU devices
        self.train_gpu = [0, 1]
        self.devices = [0, 1]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.e_layers = args.e_layers
        self.d_layers = args.d_layers
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.n_heads = args.n_heads
        self.info = args.info

        # Get the current time for result saving
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Data paths based on dataset
        if self.args.data_name == 'ETTh1':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTh1.csv'
        elif self.args.data_name == 'ETTh2':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTh2.csv'
        elif self.args.data_name == 'ETTm1':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTm1.csv'
        elif self.args.data_name == 'ETTm2':
            self.date_path = './datasets/ETT-small/china.csv'
            self.data_path = './datasets/ETT-small/ETTm2.csv'
        elif self.args.data_name == 'traffic':
            self.date_path = './datasets/traffic/usa_sanfrancisco.csv'
            self.data_path = './datasets/traffic/traffic.csv'
        elif self.args.data_name == 'electricity':
            self.date_path = './datasets/electricity/portugal.csv'
            self.data_path = './datasets/electricity/electricity.csv'
        elif self.args.data_name == 'exchange':
            self.date_path = None
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'
        elif self.args.data_name == 'illness':
            self.date_path = None
            self.data_path = './datasets/illness/national_illness.csv'

        self._get_data()
        self._get_model()

    def _get_data(self):
        # Fetch data based on dataset, adjust `get_data` and `MyDataset` functions accordingly
        train, valid, test, mean, scale, dim = get_data(self.data_path, self.date_path, args=self.args)

        self.mean = mean
        self.scale = scale
        self.args.data_dim = dim

        # Create datasets
        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        # Create DataLoaders
        self.trainloader = DataLoader(trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if self.verbose:
            print(f'train: {len(trainset)}, valid: {len(validset)}, test: {len(testset)}')

    def _get_model(self):
        # Model selection based on model name
        ngpus_per_node = len(self.train_gpu)
        print(f'Number of devices: {ngpus_per_node}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.output_path = self.output_path

        print(f'Using device: {self.device}')

        # Model initialization based on selected model
        if 'D2V_Fourier' in self.model_name:
            if self.args.data_name == 'exchange':
                self.args.mark_index = [1, 2]
            if self.args.data_name == 'ETTh1':
                self.args.mark_index = [0, 2, 4, 6, 7, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26]
            if self.args.data_name == 'traffic':
                self.args.mark_index = [0, 1, 2, 6, 7, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 26]

        # Initialize model based on model name
        if self.model_name == 'T2V_Transformer':
            self.model = T2V_Transformer(self.args)
        elif self.model_name == 'T2V_ITransformer':
            self.model = T2V_iTransformer(self.args)
        elif self.model_name == 'T2V_PatchTST':
            self.model = T2V_PatchTST(self.args)

        # D2V Models
        if self.model_name == 'D2V_Fourier_Transformer':
            self.model = D2V_Fourier_Transformer(self.args)
        elif self.model_name == 'D2V_Fourier_PatchTST':
            self.model = D2V_Fourier_PatchTST(self.args)
        elif self.model_name == 'D2V_Fourier_ITransformer':
            self.model = D2V_Fourier_iTransformer(self.args)
        elif self.model_name == 'D2V_Autoformer':
            self.model = D2V_Autoformer(self.args)
        elif self.model_name == 'D2V_Fedformer':
            self.model = D2V_Fedformer(self.args)

        # GLAFF Models
        if self.model_name == 'GLAFF_ITransformer':
            self.model = GLAFF_iTransformer(self.args)
        elif self.model_name == 'GLAFF_PatchTST':
            self.model = GLAFF_PatchTST(self.args)
        elif self.model_name == 'GLAFF_Transformer':
            self.model = GLAFF_Transformer(self.args)

        # Standard Transformer Models
        if self.model_name == 'Transformer':
            self.model = Transformer(self.args)
        elif self.model_name == 'ITransformer':
            self.model = iTransformer(self.args)
        elif self.model_name == 'PatchTST':
            self.model = PatchTST(self.args)

        # Multi-GPU training setup
        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)
            print(f'Training on GPUs: {self.devices}')

        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        # Early stopping setup
        path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler, patience=self.patience,
                                             verbose=self.verbose, path=path)

        # Loss function setup
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)
        else:
            self.criterion = nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if mode == 'test':
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, 'test')
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mode=None)

        loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
        return outputs, loss

    def train(self):
        save_manager = utils.save.SaveManager(self.args.output_dir, self.args.model_name, 'mse', compare_type='lt', ckpt_save_freq=30)
        save_manager.save_hparam(self.args)

        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # Validation and testing
            self.model.eval()
            valid_loss, test_loss = [], []
            with torch.no_grad():
                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                    valid_loss.append(loss.item())

                for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')
                    test_loss.append(loss.item())

            avg_train_loss, avg_valid_loss, avg_test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print(f"Epoch: {e+1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            self.early_stopping(avg_valid_loss, self.model, e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        output_path = os.path.join(self.output_path, self.args.data_name + '_best_model.pkl')
        self.load_best_model(path=output_path, args=self.args)

    def load_best_model(self, path, args=None):
        if not os.path.exists(path):
            print(f"Path {path} does not exist, initializing model parameters randomly.")
        else:
            ckpt = torch.load(path)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']

    def test(self):
        if self.args.resume:
            self.load_best_model(path=self.resume_path, args=self.args)

        start = datetime.now()
        self.model.eval()

        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.pred_len:, :])

        end = datetime.now()
        test_cost_time = (end - start).total_seconds()
        print(f"Test completed in {test_cost_time:.2f} seconds")

        mape_error = np.mean(self.mean) * 0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        # Reverting normalization
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean

        mape = np.mean(np.abs(dstand_trues - dstand_preds) / (dstand_trues + mape_error))

        print(f"Test: MSE: {mse:.4f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")

        np.save(os.path.join(self.output_path, self.args.data_name + '_preds'), preds)
        np.save(os.path.join(self.output_path, self.args.data_name + '_trues'), trues)

        # Log results
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/new_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR', 'epoches', 'batch_size', 'seed', 'best_mae', 'mse', 'mape',
                           'seq_len', 'label_len', 'pred_len', 'test_cost_time', 'output_path', 'info']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.now().strftime('%Y%m%d-%H%M%S')
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time, 'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size, 'seed': self.seed, 'best_mae': mae,
                  'mse': mse, 'mape': mape, 'seq_len': self.seq_len, 'label_len': self.label_len, 'pred_len': self.pred_len,
                  'test_cost_time': test_cost_time, 'output_path': self.output_path, 'info': self.info}]
        write_csv_dict(log_path, a_log, 'a+')
