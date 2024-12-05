# -*- coding:utf-8 -*-
import torch
from collections import OrderedDict
from models.unified.model import SynthSleepNetEncoder
from models.neuronet.model import NeuroNetEncoder
from peft import get_peft_model, LoraConfig


def load_pretrained_model(ckpt_path):
    # Load SynthSleepNet w/o Decoder
    def get_find_parameter(model_state, find_name):
        param_dict = OrderedDict()
        for name_, param_ in model_state.items():
            if name_.find(find_name) != -1:
                param_dict[name_] = param_
        return param_dict

    def change_parameter(pretrained_model_state, encoder_model, find_name):
        old_param = get_find_parameter(model_state=pretrained_model_state, find_name=find_name)
        new_param = encoder_model.state_dict()
        new_param = {on: nv for on, nv in zip(new_param.keys(), old_param.values())}
        return new_param

    ckpt = torch.load(ckpt_path, map_location='cpu')
    ch_names = ckpt['ch_names']
    lora_parameter = ckpt['lora_parameter']
    unimodal_parameter, multimodal_parameter = ckpt['unimodal_parameter'], ckpt['multimodal_parameter']
    multimodal_model_state = ckpt['multimodal_model_state']

    # 1. Load Unimodal (= NeuroNet) Pretrained Model
    neuronet_pretrained_model = {}
    for ch_name in ch_names:
        neuronet = NeuroNetEncoder(**unimodal_parameter)
        peft_config = LoraConfig(
            r=lora_parameter['lora_r'], lora_alpha=lora_parameter['lora_alpha'],
            lora_dropout=lora_parameter['lora_dropout'], bias='none',
            use_rslora=True, init_lora_weights='gaussian',
            target_modules=['attn.proj'],
        )
        neuronet = get_peft_model(model=neuronet, peft_config=peft_config)
        neuronet_pretrained_model[ch_name] = neuronet

    # 2. Load Multimodal Pretrained Model
    backbone = SynthSleepNetEncoder(
        backbone_networks=neuronet_pretrained_model,
        backbone_embed_dim=multimodal_parameter['backbone_embed_dim'],
        num_backbone_frames=multimodal_parameter['num_backbone_frames'],
        encoder_embed_dim=multimodal_parameter['encoder_embed_dim'],
        encoder_heads=multimodal_parameter['encoder_heads'],
        encoder_depths=multimodal_parameter['encoder_depths']
    )
    backbone.backbone_networks.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.backbone_networks,
        find_name='backbone_networks'
    ))
    backbone.backbone_embedded.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.backbone_embedded,
        find_name='backbone_embedded'
    ))
    backbone.modal_token_dict.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.modal_token_dict,
        find_name='modal_token_dict'
    ))
    backbone.multimodal_encoder_block.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_encoder_block,
        find_name='multimodal_encoder_block'
    ))
    backbone.multimodal_encoder_norm.load_state_dict(change_parameter(
        pretrained_model_state=multimodal_model_state,
        encoder_model=backbone.multimodal_encoder_norm,
        find_name='multimodal_encoder_norm'
    ))

    # 3. Freeze Parameter
    for name, module in backbone.named_modules():
        for param in module.parameters():
            param.requires_grad = False
    return backbone, ch_names, multimodal_parameter['encoder_embed_dim']
