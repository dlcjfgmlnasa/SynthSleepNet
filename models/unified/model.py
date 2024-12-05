# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, List
from timm.models.vision_transformer import Block
from models.utils import get_2d_sincos_pos_embed_flexible
from models.loss import NTXentLoss
from functools import partial
from einops.layers.torch import Rearrange


class SynthSleepNet(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List[int], temperature: float):
        super().__init__()
        self.modal_names = list(backbone_networks.keys())
        self.backbone_networks = backbone_networks
        self.modal_count, self.num_backbone_frames = len(self.backbone_networks), num_backbone_frames
        self.backbone_embed_dim = backbone_embed_dim
        self.encoder_embed_dim, self.decoder_embed_dim = encoder_embed_dim, decoder_embed_dim

        self.input_size = (self.num_backbone_frames, self.encoder_embed_dim)
        self.patch_size = (1, self.encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w
        self.mlp_ratio = 4.

        # [BackBone Network]
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.backbone_embedded = nn.ModuleDict({
            modal_name: nn.Sequential(nn.Linear(backbone_embed_dim, encoder_embed_dim),
                                      Rearrange('b t e -> b e t'),
                                      nn.BatchNorm1d(encoder_embed_dim),
                                      nn.ELU(),
                                      Rearrange('b e t -> b t e'),
                                      nn.Linear(encoder_embed_dim, encoder_embed_dim))
            for modal_name in self.modal_names
        })
        self.modal_token_dict = nn.ParameterDict({
            modal_name: nn.Parameter(torch.zeros(1, num_backbone_frames, encoder_embed_dim))
            for modal_name in self.modal_names
        })

        # [MultiModal Encoder]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.multimodal_encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_embed_dim),
                                                         requires_grad=False)
        self.multimodal_encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.multimodal_encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # [MultiModal Decoder]
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.multimodal_decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim),
                                                         requires_grad=False)
        self.multimodal_decoder_block_dict = nn.ModuleDict({
            modal_name: nn.ModuleList([
                Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6))
                for _ in range(decoder_depths)
            ])
            for modal_name in self.modal_names
        })
        self.multimodal_decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.multimodal_decoder_pred_dict = nn.ModuleDict({
            modal_name: nn.Linear(decoder_embed_dim, backbone_embed_dim, bias=True)
            for modal_name in self.modal_names
        })

        # [Contrastive Learning]
        self.backbone_projector_dict = nn.ModuleDict({
            modal_name: self.get_projection_layer([backbone_embed_dim] + projection_hidden)
            for modal_name in self.modal_names
        })
        self.fusion_projector = self.get_projection_layer([encoder_embed_dim] + projection_hidden)
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.initialize_weights()

    @staticmethod
    def get_projection_layer(projection_hidden):
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
            else:
                projectors.append(nn.Linear(h1, h2))
        return nn.Sequential(*projectors)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        multimodal_encoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.multimodal_encoder_pos_embed.shape[-1],
                                                                        (self.grid_h, self.grid_w), cls_token=False)
        self.multimodal_encoder_pos_embed.data.copy_(torch.from_numpy(multimodal_encoder_pos_embed).float().unsqueeze(0))
        multimodal_decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.multimodal_decoder_pos_embed.shape[-1],
                                                                        (self.grid_h, self.grid_w), cls_token=False)
        self.multimodal_decoder_pos_embed.data.copy_(torch.from_numpy(multimodal_decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        for model_name, modal_token in self.modal_token_dict.items():
            self.modal_token_dict[model_name] = torch.nn.init.normal_(modal_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, data, mask_ratio: float = 0.8):
        total_x = []
        unimodal_token_dict = {modal_name: None for modal_name in self.modal_names}
        mask_dict = {modal_name: None for modal_name in self.modal_names}
        ids_restore_dict = {modal_name: None for modal_name in self.modal_names}

        for unimodal_name, unimodal_x in data.items():
            encoder_out = self.backbone_networks[unimodal_name](unimodal_x)
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out)

            # add positional_encoding, modal_token
            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.multimodal_encoder_pos_embed

            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

            # add unimodal_token_dict & mask_dict & ids_restore_dict
            unimodal_token_dict[unimodal_name] = encoder_out
            mask_dict[unimodal_name] = mask
            ids_restore_dict[unimodal_name] = ids_restore
            total_x.append(x)

        # concatenation vector tokens
        x = torch.cat(total_x, dim=1)

        # apply Transformer blocks
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)
        return (x, mask_dict, ids_restore_dict), unimodal_token_dict

    def forward_decoder(self, data, ids_restores):
        split_size = (data.shape[1] - 1) // self.modal_count
        total_x = []
        for i, (modal_name, ids_restore) in enumerate(ids_restores.items()):
            start, end = i, i+split_size
            x = data[:, start:end, :]
            x = self.decoder_embed(x)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            # add pos embed
            x = x + self.multimodal_decoder_pos_embed

            # apply Transformer blocks
            blocks = self.multimodal_decoder_block_dict[modal_name]
            for block in blocks:
                x = block(x)

            x = self.multimodal_decoder_norm(x)
            x = self.multimodal_decoder_pred_dict[modal_name](x)
            total_x.append(x)

        # concatenation vector tokens
        x = torch.cat(total_x, dim=1)
        return x

    def forward(self, data, mask_ratio: float = 0.8):
        # 1. Masked Prediction
        (fusion_tokens, masks, ids_restores), unimodal_token_dict = self.forward_encoder(data, mask_ratio=mask_ratio)
        real_tokens = torch.cat([unimodal_token[:, 1:, ] for unimodal_token in unimodal_token_dict.values()], dim=1)
        pred_tokens = self.forward_decoder(fusion_tokens, ids_restores)
        mask_tokens = torch.cat([mask for mask in masks.values()], dim=-1)
        reconstruction_loss = self.forward_mae_loss(real_tokens, pred_tokens, mask_tokens)

        # 2. Contrastive Learning
        cross_contrastive_loss, cross_contrastive_accuracy = [], []
        fusion_token = torch.mean(fusion_tokens, dim=1)
        o1 = self.fusion_projector(fusion_token)

        for unimodal_name, unimodal_tokens in unimodal_token_dict.items():
            unimodal_token = torch.mean(unimodal_tokens, dim=1)
            o2 = self.backbone_projector_dict[unimodal_name](unimodal_token)
            contra_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            cross_contrastive_loss.append(contra_loss)
            cross_contrastive_accuracy.append(
                torch.mean(
                    torch.eq(torch.argmax(logits, dim=-1), labels).to(torch.float32)
                )
            )
        cross_contrastive_loss = torch.stack(cross_contrastive_loss, dim=-1)
        cross_contrastive_loss = torch.mean(cross_contrastive_loss, dim=-1)

        cross_contrastive_accuracy = torch.stack(cross_contrastive_accuracy, dim=-1)
        cross_contrastive_accuracy = torch.mean(cross_contrastive_accuracy, dim=-1)

        return reconstruction_loss, cross_contrastive_loss, cross_contrastive_accuracy

    def forward_multimodal_embed(self, data):
        (fusion_tokens, masks, ids_restores), unimodal_token_dict = self.forward_encoder(data, mask_ratio=0.0)
        fusion_token = torch.mean(fusion_tokens, dim=1)
        return fusion_token

    @staticmethod
    def forward_mae_loss(real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):
        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


class SynthSleepNetEncoder(nn.Module):
    def __init__(self,
                 backbone_networks: Dict[str, nn.Module],
                 backbone_embed_dim: int, num_backbone_frames: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int):
        super().__init__()
        self.modal_names = list(backbone_networks.keys())
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.modal_count, self.num_backbone_frames = len(self.backbone_networks), num_backbone_frames
        self.backbone_embed_dim, self.encoder_embed_dim = backbone_embed_dim, encoder_embed_dim

        self.input_size = (self.num_backbone_frames, self.encoder_embed_dim)
        self.patch_size = (1, self.encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w
        self.mlp_ratio = 4.

        # [BackBone Network]
        self.backbone_networks = nn.ModuleDict(backbone_networks)
        self.backbone_embedded = nn.ModuleDict({
            modal_name: nn.Sequential(nn.Linear(backbone_embed_dim, encoder_embed_dim),
                                      Rearrange('b t e -> b e t'),
                                      nn.BatchNorm1d(encoder_embed_dim),
                                      nn.ELU(),
                                      Rearrange('b e t -> b t e'),
                                      nn.Linear(encoder_embed_dim, encoder_embed_dim))
            for modal_name in self.modal_names
        })
        self.modal_token_dict = nn.ParameterDict({
            modal_name: nn.Parameter(torch.zeros(1, num_backbone_frames, encoder_embed_dim))
            for modal_name in self.modal_names
        })

        # [MultiModal Encoder]
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_embed_dim),
                                      requires_grad=False)
        self.multimodal_encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.multimodal_encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialization
        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     (self.grid_h, self.grid_w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, data, fusion: bool = True):
        total_x = []
        # "encoder_emb" + "positional_encoding" + "modal_token"
        for unimodal_name, unimodal_x in data.items():
            encoder_out = self.backbone_networks[unimodal_name](unimodal_x)
            encoder_emb = self.backbone_embedded[unimodal_name](encoder_out)

            x = encoder_emb[:, 1:, :] + self.modal_token_dict[unimodal_name] + self.pos_embed
            total_x.append(x)

        # concatenation vector tokens
        x = torch.cat(total_x, dim=1)

        # apply multimodal Transformer blocks
        for block in self.multimodal_encoder_block:
            x = block(x)
        x = self.multimodal_encoder_norm(x)

        # fusion tokens
        if fusion:
            x = torch.mean(x, dim=1)
        return x
