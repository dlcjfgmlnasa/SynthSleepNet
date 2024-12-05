# -*- coding:utf-8 -*-
import os
import time
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


class TorchSSLDataset(Dataset):
    def __init__(self, paths, ch_names, num_cached_parquet=5, sfreq: int = 100, shuffle=True):
        super().__init__()
        self.paths = paths
        self.num_cached_parquet = num_cached_parquet
        self.steps_cache = int(np.ceil(len(self.paths) / self.num_cached_parquet))
        self.current_parquet_idx = 0
        self.current_pd_parquets = None         # cached parquets
        self.current_indices_in_cache = []      # data index in cached parquet
        self.steps_per_epoch = 0
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.total_len = self.get_total_length()
        self.shuffle = shuffle
        self._cache_setting()

    def _cache_setting(self):
        cur_pd, cur_indices = self._cache_parquet(self.current_parquet_idx)
        self.current_pd_parquets = cur_pd
        self.current_indices_in_cache = cur_indices

    def _cache_parquet(self, idx):
        next_idx = (idx+1) * self.num_cached_parquet
        next_idx = None if next_idx > len(self.paths) else next_idx

        list_part_paths = self.paths[idx*self.num_cached_parquet:next_idx]
        sample_df = []
        for path in list_part_paths:
            series = pl.scan_parquet(os.path.join(path, 'x.parquet'))
            exist_ch_names = [ch_name for ch_name in self.ch_names if ch_name in series.columns]
            for ch_name in exist_ch_names:
                series = pl.read_parquet(os.path.join(path, 'x.parquet'))[ch_name]
                sample_df.append(series)
        sample_df = pl.concat(sample_df).rename('Signal')
        sample_numpy = sample_df.to_numpy().reshape(-1, self.sfreq * 30)

        now = time.time()
        seed = int((now - int(now)) * 100000)
        rng = np.random.RandomState(seed=seed)
        np_indices = rng.permutation(len(sample_numpy)) \
            if self.shuffle else np.arange(len(sample_numpy))
        list_indices = np_indices.tolist()

        return sample_numpy, list_indices

    def get_total_length(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths):
            series = pl.scan_parquet(os.path.join(path, 'x.parquet'))
            exist_ch_names = [ch_name for ch_name in self.ch_names if ch_name in series.columns]
            count = len(exist_ch_names)
            series = series.select(exist_ch_names[0]).collect().count() // (100 * 30)
            total_len += series.to_numpy().squeeze() * count
            del series
        return total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        refresh_idx = 1
        if len(self.current_indices_in_cache) < refresh_idx:
            self.current_parquet_idx += 1
            if self.current_parquet_idx >= self.steps_cache:
                self.current_parquet_idx = 0
            self._cache_setting()

        pd_idx = self.current_indices_in_cache.pop()
        x_raw = self.current_pd_parquets[pd_idx]
        x_raw = torch.tensor(x_raw, dtype=torch.float32)
        return x_raw


class TorchDataset(Dataset):
    def     __init__(self, paths, ch_names, event_names, sfreq: int = 100, undersampling=False):
        super().__init__()
        self.paths = paths
        self.sfreq = sfreq
        self.ch_names, self.event_names = ch_names, event_names
        self.undersampling = undersampling
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return len(self.total_y)

    def get_data(self):
        tx_df, ty_df = [], []
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))[self.event_names]
            y_series = y_series.apply(lambda t: 0 if np.sum(t) == 0 else 1)
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

        if self.undersampling:
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=777)
            tx_numpy, ty_numpy = rus.fit_resample(tx_numpy, ty_numpy)

        return tx_numpy, ty_numpy

    def __getitem__(self, item):
        x = torch.tensor(self.total_x[item], dtype=torch.float32)
        y = torch.tensor(self.total_y[item], dtype=torch.int32)
        return x, y

