# -*- coding:utf-8 -*-
import os
import tqdm
import torch
import random
import numpy as np
import polars as pl
from torch.utils.data import Dataset
import warnings


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, ch_names, sfreq: int = 100):
        super().__init__()
        self.paths = self.get_filter_paths(paths, ch_names)
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return self.total_x.shape[0]

    def get_data(self):
        tx_df, ty_df = [], []
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))[self.ch_names]
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))
            tx_df.append(x_series)
            ty_df.append(y_series)

        tx_df = pl.concat(tx_df)
        ty_df = pl.concat(ty_df)
        total_x = []
        for ch_name in self.ch_names:
            sample_x = tx_df[ch_name].to_numpy().reshape(-1, 30 * 100).squeeze()
            total_x.append(sample_x)
        total_x = np.stack(total_x, axis=1)
        total_y = ty_df.to_numpy().squeeze()
        return total_x, total_y

    @staticmethod
    def get_filter_paths(paths, ch_names):
        filter_paths = []
        ch_names = set(ch_names)
        for path in paths:
            series = pl.scan_parquet(os.path.join(path, 'x.parquet'))
            series_ch_names = set(series.columns)
            if len(ch_names - series_ch_names) == 0:
                filter_paths.append(path)
        return filter_paths

    def __getitem__(self, item):
        batch_x = torch.tensor(self.total_x[item], dtype=torch.float32)
        batch_y = torch.tensor(self.total_y[item], dtype=torch.int32)
        return batch_x, batch_y

