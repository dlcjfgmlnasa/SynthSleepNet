# -*- coding:utf-8 -*-
import os


def group_cross_validation(base_path, test_size=0.25, holdout_subject_size: int = 10):
    paths = [os.path.join(base_path, path) for path in os.listdir(base_path)]
    size = len(paths)
    train_paths, eval_paths = paths[:int(size * (1 - test_size))], paths[int(size * (1 - test_size)):]
    train_paths, val_paths = train_paths[:len(train_paths) - holdout_subject_size], \
                             train_paths[len(train_paths) - holdout_subject_size:],

    print('[K-Group Cross Validation]')
    print('   >> Train Subject Size : {}'.format(len(train_paths)))
    print('   >> Validation Subject Size : {}'.format(len(val_paths)))
    print('   >> Evaluation Subject Size : {}'.format(len(eval_paths)))

    return {'train_paths': train_paths, 'val_paths': val_paths, 'eval_paths': eval_paths}

