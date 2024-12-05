# -*- coding:utf-8 -*-
import argparse

import torch.nn as nn
import torch.optim as opt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset.utils import group_cross_validation
from downstream.linear_probing.data_loader import *
from downstream.linear_probing.model import Model
from models.utils import model_size

warnings.filterwarnings(action='ignore')

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Pretrained Checkpoint Hyperparameter
    parser.add_argument('--pretrain_ckpt_path',
                        default=os.path.join('..', '..', 'ckpt', 'multimodal',
                                             'EEG1_EOG1_ECG1', 'model', 'best_model.pth'),
                        type=str)
    parser.add_argument('--task_name', default='sleep_stage', choices=['sleep_stage', 'apnea', 'hypopnea'], type=str)
    parser.add_argument('--class_num', default=5, type=int)

    # Dataset Hyperparameter
    parser.add_argument('--base_path', default=os.path.join('..', '..', 'data', 'shhs1'))
    parser.add_argument('--holdout_subject_size', default=50, type=int)
    parser.add_argument('--test_size', default=0.10, type=float)
    parser.add_argument('--sampling', default=0.05, type=float, choices=[0.01, 0.05, 1.0])

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--train_base_learning_rate', default=0.0001, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)

    # Checkpoint Hyperparameter
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', 'ckpt', 'multimodal', 'EEG1_EOG1_ECG1'),
                        type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.train_paths, self.eval_paths = self.data_paths()
        self.model = Model(pretrain_ckpt_path=self.args.pretrain_ckpt_path,
                           class_num=self.args.class_num).to(device)
        self.ch_names = self.model.ch_names

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.clipping_norm_value = 2

        print('[Model Parameter]')
        print('   >> Task Name : {0}'.format(self.args.task_name))
        print('   >> Channel Name List : {0}'.format(', '.join(self.ch_names)))
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def train(self):
        train_dataloader, eval_dataloader = self.get_dataloader()
        best_model_state, best_score, best_result = None, 0.0, {}
        for epoch in range(self.args.train_epochs):
            # 1. Train
            step = 0
            self.model.backbone.eval()
            self.model.fc.train()
            self.optimizer.zero_grad()

            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                x = {ch_name: x[:, i, :].squeeze() for i, ch_name in enumerate(self.ch_names)}
                out = self.model(x)
                loss, _ = self.get_loss_and_performance(out, y)
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                step += 1

            # 2. Test
            test_pred, test_real = [], []
            self.model.backbone.eval()
            self.model.fc.eval()
            for x, y in eval_dataloader:
                with torch.no_grad():
                    x, y = x.to(device), y.to(device)
                    x = {ch_name: x[:, i, :].squeeze() for i, ch_name in enumerate(self.ch_names)}
                    out = self.model(x)
                    _, (pred, real) = self.get_loss_and_performance(out, y)
                    test_pred.extend(pred)
                    test_real.extend(real)

            # 3. Evaluation
            accuracy, macro_f1 = accuracy_score(y_true=test_real, y_pred=test_pred), \
                                 f1_score(y_true=test_real, y_pred=test_pred, average='macro')
            print('[Epoch] : {0:03d} \t [Accuracy] : {1:02.2f} \t [Macro-F1] : {2:02.2f}'.format(epoch,
                                                                                                 accuracy * 100,
                                                                                                 macro_f1 * 100))
            if macro_f1 > best_score:
                best_score = macro_f1
                best_model_state = self.model.state_dict()
                best_result = {'real': test_real, 'pred': test_pred}

        self.save_ckpt(model_state=best_model_state, result=best_result)

    def get_loss_and_performance(self, pred, real):
        loss = self.criterion(pred, real)
        pred = list(torch.argmax(pred, dim=-1).detach().cpu().numpy())
        real = list(real.detach().cpu().numpy())
        return loss, (pred, real)

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        train_paths, eval_paths = paths['val_paths'], paths['eval_paths']
        return train_paths, eval_paths

    def save_ckpt(self, model_state, result):
        if self.args.sampling == 1.0:
            ckpt_path = os.path.join(self.args.ckpt_path, 'linear_prob', self.args.task_name)
        else:
            ckpt_path = os.path.join(self.args.ckpt_path, 'linear_prob',
                                     self.args.task_name + '_{}'.format(self.args.sampling))
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'MultiModal_Linear_Probing',
            'model_parameter': {'pretrain_ckpt_path': os.path.abspath(self.args.pretrain_ckpt_path),
                                'class_num': self.args.class_num},
            'model_state': model_state,
            'result': result
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def get_dataloader(self):
        if self.args.task_name == 'sleep_stage':
            train_dataset, eval_dataset = SleepStageDataset(paths=self.train_paths, ch_names=self.ch_names,
                                                            sampling=self.args.sampling), \
                                          SleepStageDataset(paths=self.eval_paths, ch_names=self.ch_names)
            train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           shuffle=True), \
                                                DataLoader(eval_dataset, batch_size=self.args.train_batch_size)
            return train_dataloader, eval_dataloader

        if self.args.task_name == 'apnea':
            train_dataset, eval_dataset = ApneaDataset(paths=self.eval_paths, ch_names=self.ch_names,
                                                       sampling=self.args.sampling), \
                                          ApneaDataset(paths=self.eval_paths, ch_names=self.ch_names)
            train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           shuffle=True), \
                                                DataLoader(eval_dataset, batch_size=self.args.train_batch_size)
            return train_dataloader, eval_dataloader

        if self.args.task_name == 'hypopnea':
            train_dataset, eval_dataset = HypopneaDataset(paths=self.eval_paths, ch_names=self.ch_names,
                                                          sampling=self.args.sampling), \
                                          HypopneaDataset(paths=self.eval_paths, ch_names=self.ch_names)
            train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size,
                                                           shuffle=True), \
                                                DataLoader(train_dataset, batch_size=self.args.train_batch_size)
            return train_dataloader, eval_dataloader


if __name__ == '__main__':
    augments = get_args()
    Trainer(augments).train()
