import os
import time
import torch
import functools
import yaml
import argparse

from omegaconf import OmegaConf
from math import ceil
from pathlib import Path
from torch.utils.data import TensorDataset

from .sde import VESDE, VPSDE, DDPM, _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES
from .ema import ExponentialMovingAverage
from ..third_party_models import OpenAiUNetModel


def get_standard_score(config, sde, use_ema, load_model=True):
    score = OpenAiUNetModel(
        image_size=config.data.im_size,
        in_channels=config.model.in_channels,
        model_channels=config.model.model_channels,
        out_channels=config.model.out_channels,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        marginal_prob_std=sde.marginal_prob_std if any([isinstance(sde, classname) for classname in _SCORE_PRED_CLASSES]) else None,
        channel_mult=config.model.channel_mult,
        conv_resample=config.model.conv_resample,
        dims=config.model.dims,
        num_heads=config.model.num_heads,
        num_head_channels=config.model.num_head_channels,
        num_heads_upsample=config.model.num_heads_upsample,
        use_scale_shift_norm=config.model.use_scale_shift_norm,
        resblock_updown=config.model.resblock_updown,
        use_new_attention_order=config.model.use_new_attention_order,
        max_period=config.model.max_period
        )

    if config.sampling.load_model_from_path is not None and config.sampling.model_name is not None and load_model: 
        print(f'load score model from path: {config.sampling.load_model_from_path}')
         
    return score

def get_standard_sde(config):

    _sde_classname = config.sde.type.lower()
    if _sde_classname == 'vesde':
        sde = VESDE(
        sigma_min=config.sde.sigma_min, 
        sigma_max=config.sde.sigma_max
        )
    elif _sde_classname == 'vpsde':
        sde = VPSDE(
        beta_min=config.sde.beta_min, 
        beta_max=config.sde.beta_max
        )
    elif _sde_classname== 'ddpm':
        sde = DDPM(
        beta_min=config.sde.beta_min, 
        beta_max=config.sde.beta_max, 
        num_steps=config.sde.num_steps
        )
    else:
        raise NotImplementedError

    return sde

