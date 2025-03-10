import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
from utils.timefeatures import time_features
from utils.preprocessor import generate_labels_single, generate_labels_multiple_lookaheads, RobustZScoreNorm

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """
    A dataset class for time series forecasting used in DeepAries.

    The dataset is constructed from public data (e.g., from yfinance) and grouped by prediction dates.
    Each sample provided by the data loader is of shape (N, F), where:
      - N: Number of stocks (varies by market)
      - F: Number of normalized features per stock

    Args:
        root_path (str): Root directory for the dataset.
        data_path (str): CSV filename of the dataset.
        flag (str): Data split flag ('train', 'val', 'test', 'backtest', 'pred').
        valid_year (int or str): Year threshold for training data.
        test_year (int or str): Year threshold for testing data.
        size (list): [seq_len, label_len, pred_len].
        use_multi_horizon (bool): Whether to generate multi-horizon labels.
        lookaheads (list): List of horizons for multi-horizon labels.
        scale (bool): Whether to apply normalization.
        timeenc (int): Time encoding type (0 for simple, 1 for advanced).
        freq (str): Frequency for time feature encoding.
        step_size (int): Step size for step sampling.
        use_step_sampling (bool): Whether to use step sampling.
    """

    def __init__(self,
                 root_path,
                 data_path,
                 data,
                 flag='train',
                 valid_year=2020,
                 test_year=2021,
                 size=None,
                 use_multi_horizon=True,
                 lookaheads=[1, 5, 20],
                 scale=True,
                 timeenc=0,
                 freq='h',
                 step_size=None,
                 use_step_sampling=False):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.flag = flag
        self.valid_year = valid_year
        self.test_year = test_year
        self.use_multi_horizon = use_multi_horizon
        self.lookaheads = lookaheads
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.data = data

        self.step_size = step_size if step_size is not None else self.pred_len
        self.use_step_sampling = use_step_sampling

        self.__read_data__()
        if self.use_step_sampling and self.flag != 'pred':
            self.__build_indexes__()

    def __read_data__(self):
        market = os.path.basename(os.path.normpath(self.root_path))
        # Construct the file path: data/{market}_{data}_data.csv
        file_path = os.path.join("data", market, f"{market}_{self.data}_data.csv")
        df_raw = pd.read_csv(file_path)
        df_raw.fillna(0, inplace=True)
        df_raw['date'] = pd.to_datetime(df_raw['date']).dt.tz_localize(None)
        df_raw = df_raw.set_index(['date', 'tic']).sort_index()

        # # Split data by year
        df_train = df_raw[df_raw.index.get_level_values('date') < self.valid_year]
        df_val = df_raw[
            (df_raw.index.get_level_values('date') >= self.valid_year) &
            (df_raw.index.get_level_values('date') < self.test_year)
            ]
        df_test = df_raw[df_raw.index.get_level_values('date') >= self.test_year]

        if self.flag == 'train':
            df_split = df_train
        elif self.flag == 'val':
            df_split = df_val
        elif self.flag in ['test', 'backtest', 'pred']:
            df_split = df_test
        else:
            raise ValueError("flag must be one of ['train','val','test','backtest','pred']")

        # Generate labels
        if self.use_multi_horizon:
            df_labeled = generate_labels_multiple_lookaheads(df_split, self.lookaheads)
            label_cols = [f"return_ratio_{lh}" for lh in self.lookaheads]
        else:
            df_labeled = generate_labels_single(df_split, lookahead=self.pred_len)
            label_cols = ["return_ratio"]

        feature_cols = df_labeled.columns.drop(label_cols)
        if self.scale and self.flag in ['train', 'val', 'test', 'backtest']:
            self.scaler = RobustZScoreNorm(
                fit_start_time=df_train.index.get_level_values('date').min(),
                fit_end_time=df_train.index.get_level_values('date').max(),
                fields_group=feature_cols
            )
            self.scaler.fit(df_train)
            df_labeled = self.scaler(df_labeled)
        self.df = df_labeled[feature_cols]
        self.df_label = df_labeled[label_cols]

        df_date_only = self.df.reset_index()[['date']].drop_duplicates().sort_values(by='date')
        if self.timeenc == 0:
            df_date_only['month'] = df_date_only['date'].dt.month
            df_date_only['day'] = df_date_only['date'].dt.day
            df_date_only['weekday'] = df_date_only['date'].dt.weekday
            df_date_only['hour'] = df_date_only['date'].dt.hour
            self.data_stamp = df_date_only.drop(columns=['date']).values
        else:
            time_data = time_features(df_date_only['date'].values, freq=self.freq)
            self.data_stamp = time_data.transpose(1, 0)

        self.unique_dates = df_date_only['date'].values

    def __build_indexes__(self):
        max_start = len(self.unique_dates) - (self.seq_len + self.pred_len)
        if max_start < 0:
            raise ValueError("Data length is shorter than seq_len + pred_len.")
        self.indexes = list(range(0, max_start, self.step_size))

    def __len__(self):
        if self.flag == 'pred':
            return len(self.unique_dates) - self.seq_len
        else:
            if self.use_step_sampling:
                return len(self.indexes)
            else:
                return len(self.unique_dates) - (self.seq_len + self.pred_len)

    def __getitem__(self, idx):
        if self.flag == 'pred':
            return self.__getitem_pred__(idx)
        else:
            return self.__getitem_train__(idx)

    def __getitem_train__(self, idx):
        if self.use_step_sampling:
            idx = self.indexes[idx]

        start_date = self.unique_dates[idx]
        seq_end_idx = idx + self.seq_len
        if seq_end_idx >= len(self.unique_dates):
            seq_end_idx = len(self.unique_dates) - 1
        seq_end_date = self.unique_dates[seq_end_idx]

        label_start_idx = seq_end_idx - self.label_len
        if label_start_idx < 0:
            label_start_idx = 0
        label_start_date = self.unique_dates[label_start_idx]

        label_end_idx = label_start_idx + self.label_len + self.pred_len
        if label_end_idx >= len(self.unique_dates):
            label_end_idx = len(self.unique_dates) - 1
        label_end_date = self.unique_dates[label_end_idx]

        # Get input sequence data
        seq_mask = (self.df.index.get_level_values('date') >= start_date) & \
                   (self.df.index.get_level_values('date') < seq_end_date)
        seq_x = self.df[seq_mask].groupby(level='date').apply(lambda x: x.values)
        seq_x = np.stack(seq_x, axis=1)

        # Get time encodings for input sequence
        seq_stamp_mask = (self.unique_dates >= start_date) & (self.unique_dates < seq_end_date)
        seq_x_mark = self.data_stamp[seq_stamp_mask]

        # Get target sequence data
        label_mask = (self.df.index.get_level_values('date') >= label_start_date) & \
                     (self.df.index.get_level_values('date') < label_end_date)
        seq_y = self.df[label_mask].groupby(level='date').apply(lambda x: x.values)
        seq_y = np.stack(seq_y, axis=1)

        # Get time encodings for target sequence
        label_stamp_mask = (self.unique_dates >= label_start_date) & (self.unique_dates < label_end_date)
        seq_y_mark = self.data_stamp[label_stamp_mask]

        # Get label for the final date of the sequence
        final_date = self.unique_dates[seq_end_idx]
        mask_gt = (self.df_label.index.get_level_values('date') == final_date)
        if self.use_multi_horizon:
            label_cols = [f"return_ratio_{lh}" for lh in self.lookaheads]
            ground_true = self.df_label[mask_gt][label_cols].values
        else:
            ground_true = self.df_label[mask_gt]["return_ratio"].values

        return seq_x, seq_y, seq_x_mark, seq_y_mark, ground_true

    def __getitem_pred__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.seq_len
        if end_idx > len(self.unique_dates):
            end_idx = len(self.unique_dates)
        start_date = self.unique_dates[start_idx]
        end_date = self.unique_dates[end_idx - 1]

        seq_mask = (self.df.index.get_level_values('date') >= start_date) & \
                   (self.df.index.get_level_values('date') <= end_date)
        seq_x = self.df[seq_mask].groupby(level='date').apply(lambda x: x.values)
        seq_x = np.stack(seq_x, axis=1)

        seq_stamp_mask = (self.unique_dates >= start_date) & (self.unique_dates <= end_date)
        seq_x_mark = self.data_stamp[seq_stamp_mask]

        return seq_x, seq_x_mark
