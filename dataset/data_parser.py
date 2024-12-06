# -*- coding:utf-8 -*-
import os
import re
import abc
import mne
import tqdm
import warnings
import argparse
import numpy as np
import polars as pl
from typing import Dict
from scipy import signal


warnings.filterwarnings(action='ignore')
mne.set_log_level(False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',
                        default=os.path.join('..', '..', '..', '..', 'Dataset', 'SHHS', 'polysomnography', 'edfs', 'shhs1'),
                        help='Sleep Heart Health Study (SHHS) EDF file path',
                        type=str)
    parser.add_argument('--trg_path', default=os.path.join('..', 'data', 'shhs1'),
                        help='Save Path',
                        type=str)
    parser.add_argument('--type', default='shhs1', choices=['shhs1', 'shhs2'],
                        help='Cohort Type',
                        type=str)
    return parser.parse_args()


class Base(object):
    def __init__(self):
        super().__init__()
        self.r_channel_names = ['EEG_C4', 'EEG_C3', 'EOG_Left', 'EOG_Right', 'ECG', 'EMG_Chin', 'Airflow']
        self.r_event_names = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'SpO2 Desaturation']
        self.r_freqs = 100

    @abc.abstractmethod
    def bridge_event_name(self) -> Dict:
        pass

    @abc.abstractmethod
    def bridge_channel_name(self) -> Dict:
        pass

    @abc.abstractmethod
    def parser(self, path) -> (np.array, np.array):
        pass

    def preprocessing(self, ch_name, data):
        if ch_name in ['EEG_C4', 'EEG_C3']:
            info = mne.create_info(ch_names=['eeg'], ch_types='eeg', sfreq=self.r_freqs)
            scaler = mne.decoding.Scaler(info=info, scalings='median')
            data = mne.EpochsArray(np.expand_dims(data, axis=1), info=info)
            data = data.copy().filter(l_freq=0.5, h_freq=40)
            data = scaler.fit_transform(data.get_data()).squeeze()
            return data
        if ch_name in ['EOG_Left', 'EOG_Right']:
            info = mne.create_info(ch_names=['1'], ch_types='eog', sfreq=self.r_freqs)
            scaler = mne.decoding.Scaler(info=info, scalings='median')
            data = mne.EpochsArray(np.expand_dims(data, axis=1), info=info)
            data = data.copy().filter(l_freq=0.5, h_freq=40, picks='eog')
            data = scaler.fit_transform(data.get_data()).squeeze()
            return data
        if ch_name in ['ECG']:
            info = mne.create_info(ch_names=['1'], ch_types='ecg', sfreq=self.r_freqs)
            scaler = mne.decoding.Scaler(info=info, scalings='median')
            data = mne.EpochsArray(np.expand_dims(data, axis=1), info=info)
            data = data.copy().filter(l_freq=3, h_freq=30, picks='ecg')
            data = scaler.fit_transform(data.get_data()).squeeze()
            return data
        if ch_name in ['EMG_Chin']:
            info = mne.create_info(ch_names=['1'], ch_types='emg', sfreq=self.r_freqs)
            scaler = mne.decoding.Scaler(info=info, scalings='median')
            data = mne.EpochsArray(np.expand_dims(data, axis=1), info=info)
            data = data.copy().filter(l_freq=25, h_freq=49, picks='emg')
            data = scaler.fit_transform(data.get_data()).squeeze()
            return data
        if ch_name in ['Airflow']:
            info = mne.create_info(ch_names=['1'], ch_types='resp', sfreq=self.r_freqs)
            scaler = mne.decoding.Scaler(info=info, scalings='median')
            data = mne.EpochsArray(np.expand_dims(data, axis=1), info=info)
            data = data.copy().filter(l_freq=0.05, h_freq=1, picks='resp')
            data = scaler.fit_transform(data.get_data()).squeeze()
            return data

    def save(self, scr_path: str, trg_path: str):
        x, y, ch_names, ev_names = self.parser(scr_path)
        fname = os.path.basename(trg_path).split('-')[-1]
        fpath = os.path.join(os.path.dirname(trg_path), fname)

        if not os.path.exists(os.path.join(fpath)):
            os.makedirs(os.path.join(fpath))

        x_df = pl.DataFrame({ch_name: x[i].reshape(-1) for i, ch_name in enumerate(ch_names)})
        y_df = pl.DataFrame({ev_name: y[i].astype(np.int32) for i, ev_name in enumerate(ev_names)})

        x_df.write_parquet(os.path.join(fpath, 'x.parquet'))
        y_df.write_parquet(os.path.join(fpath, 'y.parquet'))


class SHHS(Base):
    def __init__(self):
        super().__init__()
        self.ch_names = self.bridge_channel_name()
        self.event_names = self.bridge_event_name()

    def bridge_channel_name(self) -> Dict:
        shhs_channel_names = ['EEG(SEC)', 'EEG', 'EOG(L)', 'EOG(R)', 'ECG', 'EMG', 'NEW AIR']
        bridge = {shhs_ch_name: ch_name for shhs_ch_name, ch_name in zip(shhs_channel_names, self.r_channel_names)}
        return bridge

    def bridge_event_name(self) -> Dict:
        shhs_event_names = ['Obstructive apnea', 'Central apnea', 'Mixed apnea', 'Hypopnea', 'SpO2 desaturation']
        bridge = {shhs_event_name: event_name for shhs_event_name, event_name
                  in zip(shhs_event_names, self.r_event_names)}
        return bridge

    def parser(self, path) -> (np.array, np.array):
        edf_data = mne.io.read_raw_edf(path, preload=True)
        ch_names = [ch_name.upper() for ch_name in edf_data.ch_names]
        o_freqs = int(edf_data.info['sfreq'])
        temp = edf_data.get_data()

        data = {}
        for ch_name1, ch_name2 in self.ch_names.items():
            try:
                idx = ch_names.index(ch_name1)
                sig = temp[idx].reshape(-1, 30 * o_freqs)
                sig = signal.resample(sig, 30 * self.r_freqs, axis=-1)
                sig = self.preprocessing(ch_name2, sig)
                data[ch_name2] = sig
            except ValueError as e:
                continue

        name_ = os.path.basename(path).split('.')[0] + '-nsrr.xml'
        event_path = os.path.join(*path.split('/')[:-3], 'annotations-events-nsrr', name_.split('-')[0], name_)
        with open(event_path, 'r') as f:
            content = f.read()

        stage = self.get_stage(content=content)
        event = self.get_event(content=content, size=len(stage))
        event['Sleep Stage'] = stage

        x, ch_name = self.packaging(content=data)
        y, ev_name = self.packaging(content=event)
        return x, y, ch_name, ev_name

    @staticmethod
    def packaging(content: Dict):
        values, labels = [], []
        for k, v in content.items():
            labels.append(k)
            values.append(v)
        values = np.stack(values, axis=0)
        labels = np.stack(labels, axis=0)
        return values, labels

    def get_event(self, content, size) -> Dict:
        event_content = re.findall(
            r'<EventType>Respiratory.Respiratory</EventType>\n' +
            r'<EventConcept>.+</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
            r'<Duration>[0-9\.]+</Duration>',
            content)

        events = {self.event_names[label]: np.zeros(size) for label in self.event_names.keys()}
        for pattern in event_content:
            lines = pattern.splitlines()
            event_name = lines[1].split('|')[0][14:]
            start = float(re.sub(r'[^0-9\.]', '', lines[2]))
            duration = float(re.sub(r'[^0-9\.]', '', lines[3]))
            if event_name not in self.event_names.keys():
                continue

            epoch_start, epoch_end = round(start / 30), round((duration + start) / 30)
            temp = events[self.event_names[event_name]]
            temp[epoch_start:epoch_end] = 1
            events[self.event_names[event_name]] = temp
        return events

    @staticmethod
    def get_stage(content) -> np.array:
        stage_label = {0: 0,    # 0: Wake
                       1: 1,    # 1: N1
                       2: 2,    # 2: N2
                       3: 3,    # 3: N3
                       4: 3,    # 4: N3
                       5: 4,    # 5: REM
                       9: 0}    # 9: Wake
        stage_content = re.findall(
            r'<EventType>Stages.Stages</EventType>\n' +
            r'<EventConcept>.+</EventConcept>\n' +
            r'<Start>[0-9\.]+</Start>\n' +
            r'<Duration>[0-9\.]+</Duration>',
            content)
        stages = []
        for pattern in stage_content:
            lines = pattern.splitlines()
            stage_line = lines[1]
            stage = int(stage_line[-16])
            duration_line = lines[3]
            duration = float(duration_line[10:-11])
            assert duration % 30 == 0.
            epochs_duration = int(duration) // 30
            stages += [stage]*epochs_duration
        stages = np.array([stage_label[stage] for stage in stages])
        return stages


if __name__ == '__main__':
    augment = get_args()
    if augment.type == 'shhs1':
        # Sleep Heart Health Study 1 Version
        # https://sleepdata.org/datasets/shhs
        src_base_path_ = augment.src_path
        trg_base_path_ = augment.trg_path
        dataset = SHHS()
        for name__ in tqdm.tqdm(os.listdir(src_base_path_)):
            try:
                src_path_ = os.path.join(src_base_path_, name__)
                trg_path_ = os.path.join(trg_base_path_, name__.split('.')[0])
                dataset.save(src_path_, trg_path_)
            except Exception as e:
                continue

    elif augment.type == 'shhs2':
        # Sleep Heart Health Study 2 Version
        # https://sleepdata.org/datasets/shhs
        src_base_path_ = augment.src_path
        trg_base_path_ = augment.trg_path
        dataset = SHHS()
        for name__ in tqdm.tqdm(os.listdir(src_base_path_)):
            try:
                src_path_ = os.path.join(src_base_path_, name__)
                trg_path_ = os.path.join(trg_base_path_, name__.split('.')[0] + '.parquet')
                dataset.save(src_path_, trg_path_)
            except Exception as e:
                continue
