from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
import utils.config
import utils.save
from datetime import datetime
from layers.Quantile_loss import *


class EXP:
    def __init__(self, args):
        assert args.resume_dir == args.output_dir

        # Output directory management
        if args.output_dir in [None, 'None', 'none']:
            args.output_dir = None
            utils.config.create_output_dir(args)  # Create output directory
            args.resume_dir = args.output_dir
        else:
            args.output_dir = os.path.join('experiments', args.output_dir)
            args.resume_dir = os.path.join('experiments', args.resume_dir)

        # Create model directory if it doesn't exist
        output_path = os.path.join(args.output_dir, args.model_name)
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

        # Handle model resume path
        resume_path = os.path.join(args.resume_dir, args.model_name)
        if not os.path.exists(resume_path):
            raise ValueError('Checkpoint path not found')
        self.resume_path = os.path.join(resume_path, f'{args.data_name}_best_model.pkl')

        # Set experiment parameters
        self._initialize_parameters(args)
        self._get_data()
        self._get_model()

    def _initialize_parameters(self, args):
        self.seq_len, self.label_len, self.pred_len = args.seq_len, args.label_len, args.pred_len
        self.batch_size, self.train_batch = args.batch_size, args.batch_size
        self.epochs, self.patience = args.epoches, args.patience
        self.lr, self.args = args.lr, args
        self.train_gpu, self.devices = [1], [0]
        self.model_name, self.data_name = args.model_name, args.data_name
        self.seed, self.d_model, self.d_ff = args.seed, args.d_model, args.d_ff
        self.e_layers, self.d_layers = args.e_layers, args.d_layers
        self.patch_len, self.stride = args.patch_len, args.stride
        self.n_heads, self.info = args.n_heads, args.info
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get current time for logging

        # Set dataset paths based on the data name
        self._set_dataset_paths()

    def _set_dataset_paths(self):
        dataset_paths = {
            'ETTh1': ('./datasets/ETT-small/china.csv', './datasets/ETT-small/ETTh1.csv'),
            'ETTh2': ('./datasets/ETT-small/china.csv', './datasets/ETT-small/ETTh2.csv'),
            'ETTm1': ('./datasets/ETT-small/china.csv', './datasets/ETT-small/ETTm1.csv'),
            'ETTm2': ('./datasets/ETT-small/china.csv', './datasets/ETT-small/ETTm2.csv'),
            'traffic': ('./datasets/traffic/usa_sanfrancisco.csv', './datasets/traffic/traffic.csv'),
            'electricity': ('./datasets/electricity/portugal.csv', './datasets/electricity/electricity.csv'),
            'exchange': (None, './datasets/exchange_rate/exchange_rate.csv'),
            'illness': (None, './datasets/illness/national_illness.csv'),
        }
        self.date_path, self.data_path = dataset_paths.get(self.args.data_name, (None, None))
        if not self.data_path:
            raise ValueError("Data path not found for the selected dataset")

    def _get_data(self):
        # Fetch data based on dataset and arguments
        train, valid, test, mean, scale, dim = get_data(self.data_path, self.date_path, args=self.args)
        self.mean, self.scale, self.args.data_dim = mean, scale, dim

        # Create dataset and dataloaders for training, validation, and testing
        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        self.trainloader = DataLoader(trainset, batch_size=self.train_batch, shuffle=True, drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        if self.verbose:
            print(f"Train: {len(trainset)}, Valid: {len(validset)}, Test: {len(testset)}")

    def _get_model(self):
        # Get model based on the model name
        ngpus_per_node = len(self.train_gpu)
        print(f"Number of devices: {ngpus_per_node}")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args.device = self.device
        self.args.output_path = self.output_path

        print(f"Using device: {self.device}")

        # Choose the model based on model name
        if 'D2Vformer' in self.model_name:
            mask_spectrum = self._get_mask_spectrum()
            self.args.mask_spectrum = mask_spectrum

        model_classes = {
            'D2Vformer': D2Vformer,
            'D2Vformer_s': D2Vformer_simple,
        }

        # Instantiate the model
        self.model = model_classes.get(self.model_name, None)(self.args)

        # Multi-GPU handling
        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)

        # Optimizer and scheduler setup
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        # Early stopping setup
        path = os.path.join(self.output_path, f"{self.args.data_name}_best_model.pkl")
        self.early_stopping = EarlyStopping(optimizer=self.optimizer, scheduler=self.scheduler,
                                            patience=self.patience, verbose=self.verbose, path=path)

        # Loss function setup
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)
        else:
            self.criterion = nn.MSELoss()

    def _get_mask_spectrum(self):
        # Get shared frequency spectrums
        train_loader = self.trainloader
        self.args.alpha = 0.2
        amps = 0.0
        for data in train_loader:
            lookback_window = data[0]
            amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

        mask_spectrum = amps.topk(int(amps.shape[0] * self.args.alpha)).indices
        return mask_spectrum

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, mode):
        # Process a single batch of data
        batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
        batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)

        # Forward pass
        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark, mode)
        loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
        return outputs, loss

    def train(self):
        # Training procedure
        self.nan_flag = False
        save_manager = utils.save.SaveManager(self.args.output_dir, self.args.model_name, 'mse', compare_type='lt',
                                              ckpt_save_freq=30)
        save_manager.save_hparam(self.args)

        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):
                if self.nan_flag:
                    print("NaN detected during training!")
                    break
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
                if torch.isnan(loss):
                    self.nan_flag = True
                    print("Loss is NaN!")
                    break
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = [self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')[1].item()
                          for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(self.validloader)]
            test_loss = [self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='val')[1].item() for
                         batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(self.testloader)]

            avg_train_loss, avg_valid_loss, avg_test_loss = np.average(train_loss), np.average(valid_loss), np.average(
                test_loss)
            print(
                f"Epoch {e + 1}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            self.early_stopping(avg_valid_loss, self.model, e)
            if self.early_stopping.early_stop:
                break

            self.scheduler.step()

        # Save the best model after training
        output_path = os.path.join(self.output_path, f"{self.args.data_name}_best_model.pkl")
        self.load_best_model(path=output_path, args=self.args)

    def load_best_model(self, path, args=None):
        # Load the best model from a checkpoint
        if not os.path.exists(path):
            print(f"Checkpoint path {path} does not exist, using random initialization")
        else:
            ckpt = torch.load(path)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.start_epoch = ckpt['epoch']

    def test(self):
        # Testing procedure
        if self.args.resume:
            self.load_best_model(path=self.resume_path, args=self.args)

        start_time = datetime.now()
        self.model.eval()

        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:, -self.pred_len:, :])

        end_time = datetime.now()
        test_time = (end_time - start_time).total_seconds()
        print(f"Testing took {test_time:.2f} seconds")

        # Calculate metrics
        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        dstand_preds = preds * self.scale + self.mean
        dstand_trues = trues * self.scale + self.mean
        mape = np.mean(np.abs(dstand_trues - dstand_preds) / (dstand_trues + np.mean(self.mean) * 0.1))

        print(f"Test: MSE: {mse:.4f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")

        # Save predictions and true values
        np.save(os.path.join(self.output_path, f"{self.args.data_name}_preds"), preds)
        np.save(os.path.join(self.output_path, f"{self.args.data_name}_trues"), trues)

        # Log the results
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')
        log_path = './results/new_experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR', 'epoches', 'batch_size', 'seed', 'best_mae', 'mse', 'mape',
                           'seq_len', 'label_len', 'pred_len', 'test_cost_time', 'output_path', 'info']]
            write_csv(log_path, table_head, 'w+')

        # Log the test results in CSV
        time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_entry = [
            {'dataset': self.data_name, 'model': self.model_name, 'time': time, 'LR': self.lr, 'epoches': self.epochs,
             'batch_size': self.batch_size, 'seed': self.seed, 'best_mae': mae, 'mse': mse, 'mape': mape,
             'seq_len': self.seq_len, 'label_len': self.label_len, 'pred_len': self.pred_len,
             'test_cost_time': test_time, 'output_path': self.output_path, 'info': self.info}]
        write_csv_dict(log_path, log_entry, 'a+')
