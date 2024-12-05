# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from downstream.utils import load_pretrained_model
from peft import get_peft_model, LoraConfig


class Model(nn.Module):
    def __init__(self, pretrain_ckpt_path, temporal_context_length, class_num):
        super().__init__()
        self.backbone, self.ch_names, self.backbone_embed_dim = load_pretrained_model(pretrain_ckpt_path)
        self.backbone = get_peft_model(
            model=self.backbone,
            peft_config=LoraConfig(
                r=4, lora_alpha=8, lora_dropout=0.05, bias='none',
                use_rslora=True, init_lora_weights='gaussian',
                target_modules=[
                    'base_model.model.frame_backbone.feature_layer.0',
                    'base_model.model.frame_backbone.feature_layer.2',
                    'multimodal_encoder_block.3.mlp.fc1',
                    'multimodal_encoder_block.3.mlp.fc2',
                ]
            )
        )
        self.temporal_context_length = temporal_context_length
        self.hidden_dim = self.backbone_embed_dim // 2
        self.mamba = Mamba2(d_model=self.backbone_embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.BatchNorm1d(self.backbone_embed_dim)
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
        seq_latent = []
        for i in range(self.temporal_context_length):
            sample_dict = {ch_name: x_[:, i, :] for ch_name, x_ in x.items()}
            latent = self.backbone(sample_dict)
            latent = self.norm(latent)
            seq_latent.append(latent)

        seq_latent1 = torch.stack(seq_latent, dim=1)
        seq_latent2 = torch.stack(seq_latent, dim=1)
        seq_latent2 = self.mamba(seq_latent2)

        out = []
        for tokens in torch.split(seq_latent1 + seq_latent2, split_size_or_sections=1, dim=1):
            o = self.fc(tokens.squeeze())
            out.append(o)
        out = torch.stack(out, dim=1)
        return out

