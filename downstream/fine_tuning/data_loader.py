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


def get_filter_paths(paths, ch_names):
    filter_paths = []
    ch_names = set(ch_names)
    for path in paths:
        series = pl.scan_parquet(os.path.join(path, 'x.parquet'))
        series_ch_names = set(series.columns)
        if len(ch_names - series_ch_names) == 0:
            filter_paths.append(path)
    return filter_paths


def many_to_many(elements, temporal_context_length, overlap_length):
    size = len(elements)
    total = []
    if size <= temporal_context_length:
        return elements
    for i in range(0, size-temporal_context_length+1, overlap_length):
        temp = np.array(elements[i:i+temporal_context_length, ...])
        total.append(temp)
    total.append(elements[size-temporal_context_length:size, ...])
    total = np.array(total)
    return total


class SleepStageDataset(Dataset):
    def __init__(self, paths, ch_names, temporal_context_length, overlap_length, sampling: float = 1.0):
        super().__init__()
        self.paths = get_filter_paths(paths, ch_names)
        self.ch_names = ch_names
        self.temporal_context_length, self.overlap_length = temporal_context_length, overlap_length
        self.sampling = sampling
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return len(self.total_y)

    def get_data(self):
        tx, ty = [], []
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))[self.ch_names]
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))['Sleep Stage']

            xn_series = []
            for ch_name in self.ch_names:
                sample_x = x_series[ch_name].to_numpy().reshape(-1, 30 * 100).squeeze()
                sample_x = many_to_many(sample_x, self.temporal_context_length, self.overlap_length)
                xn_series.append(sample_x)
            xn_series = np.stack(xn_series, axis=2)
            yn_series = many_to_many(y_series.to_numpy(), self.temporal_context_length, self.overlap_length)
            tx.append(xn_series)
            ty.append(yn_series)

        total_x = np.concatenate(tx, axis=0)
        total_y = np.concatenate(ty, axis=0)

        if self.sampling != 1.0:
            select_index = np.random.choice(np.arange(len(total_y)),
                                            int(len(total_y) * self.sampling), replace=True)
            total_x = total_x[select_index]
            total_y = total_y[select_index]

        return total_x, total_y

    def __getitem__(self, item):
        x = torch.tensor(self.total_x[item], dtype=torch.float32)
        y = torch.tensor(self.total_y[item], dtype=torch.long)
        return x, y


class ApneaDataset(Dataset):
    def __init__(self, paths, ch_names, temporal_context_length, overlap_length, sampling: float = 1.0):
        super().__init__()
        self.paths = get_filter_paths(paths, ch_names)
        self.ch_names = ch_names
        self.temporal_context_length, self.overlap_length = temporal_context_length, overlap_length
        self.sampling = sampling
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return len(self.total_y)

    def get_data(self):
        tx_df, ty_df = [], []
        event_names = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea']
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))[self.ch_names]
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))[event_names]
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
        total_y = np.apply_along_axis(lambda x: np.sum(x).astype(np.bool_).astype(np.int32),
                                      axis=-1, arr=total_y)
        total_x = many_to_many(total_x, self.temporal_context_length, self.overlap_length)
        total_y = many_to_many(total_y, self.temporal_context_length, self.overlap_length)
        return total_x, total_y

    def __getitem__(self, item):
        x = torch.tensor(self.total_x[item], dtype=torch.float32)
        y = torch.tensor(self.total_y[item], dtype=torch.long)
        return x, y


class HypopneaDataset(Dataset):
    def __init__(self, paths, ch_names, temporal_context_length, overlap_length, sampling: float = 1.0):
        super().__init__()
        self.paths = get_filter_paths(paths, ch_names)
        self.ch_names = ch_names
        self.temporal_context_length, self.overlap_length = temporal_context_length, overlap_length
        self.sampling = sampling
        self.total_x, self.total_y = self.get_data()

    def __len__(self):
        return len(self.total_y)

    def get_data(self):
        tx_df, ty_df = [], []
        for path in tqdm.tqdm(self.paths):
            x_series = pl.read_parquet(os.path.join(path, 'x.parquet'))[self.ch_names]
            y_series = pl.read_parquet(os.path.join(path, 'y.parquet'))['Hypopnea']
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
        total_y = np.apply_along_axis(lambda x: np.sum(x).astype(np.bool_).astype(np.int32),
                                      axis=-1, arr=total_y)
        total_x = many_to_many(total_x, self.temporal_context_length, self.overlap_length)
        total_y = many_to_many(total_y, self.temporal_context_length, self.overlap_length)
        return total_x, total_y

    def __getitem__(self, item):
        x = torch.tensor(self.total_x[item], dtype=torch.float32)
        y = torch.tensor(self.total_y[item], dtype=torch.long)
        return x, y

