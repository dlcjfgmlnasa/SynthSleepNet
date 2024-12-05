# -*- coding:utf-8 -*-
import sys
sys.path.extend(['/home/brainlab/Workspace/Chlee/MultiModal_for_Sleep',
                 '/home/brainlab/Workspace/Chlee/MultiModal_for_Sleep'])
import os
import mne
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import torch.optim as opt
from models.utils import model_size
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.utils import group_cross_validation
from pretrained.unimodal.emg.data_loader import TorchDataset
from models.neuronet.model import NeuroNet


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset Hyperparameter
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', 'data', 'shhs1'))
    parser.add_argument('--ch_names', default=['EMG_Chin'], choices=['EEG_C4', 'EEG_C3',
                                                                     'EOG_Left', 'EOG_Right',
                                                                     'ECG', 'EMG_Chin', 'Airflow'])
    parser.add_argument('--event_names', default=['Sleep Stage'])
    parser.add_argument('--holdout_subject_size', default=50, type=int)
    parser.add_argument('--sfreq', default=100, type=int)
    parser.add_argument('--test_size', default=0.10, type=float)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--train_base_learning_rate', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=256, type=int)
    parser.add_argument('--train_batch_accumulation', default=1, type=int)

    # Model Hyperparameter
    parser.add_argument('--second', default=30, type=int)
    parser.add_argument('--time_window', default=4, type=int)
    parser.add_argument('--time_step', default=1, type=int)

    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=3, type=int)

    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=20, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt', 'unimodal'), type=str)
    parser.add_argument('--ckpt_name', default='emg', type=str)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = NeuroNet(
            fs=args.sfreq, second=args.second, time_window=args.time_window, time_step=args.time_step,
            encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
            encoder_depths=args.encoder_depths,
            decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
            decoder_depths=args.decoder_depths,
            projection_hidden=args.projection_hidden, temperature=args.temperature
        ).to(device)

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.train_paths, self.val_paths, self.eval_paths = self.data_paths()
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.tensorboard_path = os.path.join(self.args.ckpt_path, self.args.ckpt_name, 'tensorboard')

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)
        self.clipping_norm_value = 2.0

        print('[NeuroNet Parameter]')
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Frame Size : {}'.format(self.model.num_patches))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def train(self):
        train_dataset = TorchDataset(paths=self.train_paths, ch_names=self.args.ch_names,
                                     event_names=self.args.event_names, sfreq=self.args.sfreq)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, drop_last=True,
                                      shuffle=True)
        val_dataset = TorchDataset(paths=self.val_paths, ch_names=[self.args.ch_names[0]],
                                   event_names=self.args.event_names, sfreq=self.args.sfreq)
        val_dataloader = DataLoader(val_dataset, batch_size=self.args.train_batch_size // 2)
        eval_dataset = TorchDataset(paths=self.eval_paths, ch_names=[self.args.ch_names[0]],
                                    event_names=self.args.event_names, sfreq=self.args.sfreq)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size // 2)

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.args.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_dataloader:
                x = x.to(device)
                out = self.model(x, mask_ratio=self.args.mask_ratio)
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out

                contrastive_loss = self.args.alpha * contrastive_loss
                loss = recon_loss + contrastive_loss
                loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm_value)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:08d}  '
                          '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                          '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                            epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                            self.compute_metrics(cl_logits, cl_labels)))

                self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                self.tensorboard_writer.add_scalar('Contrastive Loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total Loss', loss, total_step)

                step += 1
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_dataloader, eval_dataloader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            self.optimizer.step()
            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(eval_dataloader)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)

        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                latent = self.model.forward_latent(x)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)
        return total_x, total_y

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, self.args.ckpt_name, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'NeuroNet-EMG',
            'model_state': model_state,
            'model_parameter': {
                'fs': self.args.sfreq, 'second': self.args.second,
                'time_window': self.args.time_window, 'time_step': self.args.time_step,
                'encoder_embed_dim': self.args.encoder_embed_dim, 'encoder_heads': self.args.encoder_heads,
                'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim, 'decoder_heads': self.args.decoder_heads,
                'decoder_depths': self.args.decoder_depths,
                'projection_hidden': self.args.projection_hidden, 'temperature': self.args.temperature
            },
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'val_paths': self.val_paths, 'eval_paths': self.eval_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def data_paths(self):
        paths = group_cross_validation(base_path=self.args.base_path,
                                       test_size=self.args.test_size,
                                       holdout_subject_size=self.args.holdout_subject_size)
        train_paths, val_paths, eval_paths = paths['train_paths'], paths['val_paths'], paths['eval_paths']
        return train_paths, val_paths, eval_paths

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy


if __name__ == '__main__':
    augments = get_args()
    trainer = Trainer(augments)
    trainer.train()
