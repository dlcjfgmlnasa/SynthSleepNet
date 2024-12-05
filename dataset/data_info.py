# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from zipfile import BadZipFile


def get_data_info(base_path: str):
    paths = os.listdir(base_path)
    temp = {'path': [], 'ch_name': [], 'event_name': []}
    for path in paths:
        try:
            path = os.path.join(base_path, path)
            data = np.load(path)
            ch_name, ev_name = data['ch_name'], data['ev_name']
            path = os.path.abspath(path)
            ev_name = '|'.join(ev_name)
            ch_name = '|'.join(ch_name)

            temp['path'].append(path)
            temp['ch_name'].append(ch_name)
            temp['event_name'].append(ev_name)
        except BadZipFile:
            pass

    temp_df = pd.DataFrame(temp)
    temp_df.to_csv(os.path.join('..', 'shhs1.csv'), index=False)


if __name__ == '__main__':
    get_data_info(base_path=os.path.join('..', 'data', 'shhs1'))
