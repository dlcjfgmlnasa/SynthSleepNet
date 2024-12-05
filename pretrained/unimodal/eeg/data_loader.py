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
    def __init__(self, paths, ch_names, event_names, sfreq: int = 100):
        super().__init__()
        self.paths = paths
        self.sfreq = sfreq
        self.ch_names, self.event_names = ch_names, event_names
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return len(self.total_y)

    def get_data(self):
        tx_df, ty_df = [], []
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))[self.event_names]
            for ch_name in self.ch_names:
                try:
                    tx_df.append(x_series[ch_name])
                    ty_df.append(y_series)
                except pl.exceptions.ColumnNotFoundError:
                    continue

        tx_df = pl.concat(tx_df).rename('Signal')
        ty_df = pl.concat(ty_df)
        tx_numpy = tx_df.to_numpy().reshape([-1, 30 * 100]).squeeze()
        ty_numpy = ty_df.to_numpy().squeeze()
        return tx_numpy, ty_numpy

    def __getitem__(self, item):
        x = torch.tensor(self.total_x[item], dtype=torch.float32)
        y = torch.tensor(self.total_y[item], dtype=torch.int32)
        return x, y

