# -*- coding:utf-8 -*-
import torch.nn as nn
from downstream.utils import load_pretrained_model


class Model(nn.Module):
    def __init__(self, pretrain_ckpt_path, class_num):
        super().__init__()
        self.backbone, self.ch_names, self.backbone_embed_dim = load_pretrained_model(pretrain_ckpt_path)
        self.hidden_dim = self.backbone_embed_dim // 2
        self.dropout_p = 0.5
        self.fc = nn.Sequential(
            nn.Linear(self.backbone_embed_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, class_num)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
