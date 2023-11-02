''' 
Inspired to https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py 
'''
from typing import Optional, Any, Dict, Tuple
from torch import Tensor

import os
import torchvision
import numpy as np
import torch

from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .utils import _schedule_jump
from ..diffusion import SDE
from ..third_party_models import OpenAiUNetModel

class BaseSampler:
    def __init__(self, 
        score: OpenAiUNetModel, 
        sde: SDE,
        predictor: callable,
        sample_kwargs: Dict,
        device: Optional[Any] = None
        ) -> None:

        self.score = score
        self.sde = sde
        self.predictor = predictor
        self.sample_kwargs = sample_kwargs
        self.device = device
    
    def sample(self,
        c: Tensor,
        logg_kwargs: Dict = {},
        logging: bool = True
        ) -> Tensor:

        if logging:
            writer = SummaryWriter(log_dir=os.path.join(logg_kwargs['log_dir'], str(logg_kwargs['sample_num'])))
        
        num_steps = self.sample_kwargs['num_steps']
        __iter__ = None

        assert self.sde.num_steps >= num_steps
        skip = self.sde.num_steps // num_steps
        # if ``self.sample_kwargs['travel_length']'' is 1. and ''self.sample_kwargs['travel_repeat']'' is 1. 
        # ``_schedule_jump'' behaves as ``np.arange(-1. num_steps, 1)[::-1]''
        time_steps = _schedule_jump(num_steps, self.sample_kwargs['travel_length'], self.sample_kwargs['travel_repeat']) 
        time_pairs = list((i*skip, j*skip if j>0 else -1)  for i, j in zip(time_steps[:-1], time_steps[1:]))
        __iter__= time_pairs


        step_size = time_steps[0] - time_steps[1]
        init_x = self.sde.prior_sampling([c.shape[0], *self.sample_kwargs['im_shape']]).to(self.device)

        
        if logging:
            writer.add_image('init_x', torchvision.utils.make_grid(init_x, 
                normalize=True, scale_each=True), global_step=0)
            if logg_kwargs['ground_truth'] is not None: writer.add_image(
                'ground_truth', torchvision.utils.make_grid(logg_kwargs['ground_truth'].squeeze(), 
                    normalize=True, scale_each=True), global_step=0)
            writer.add_image('cond_inp', torchvision.utils.make_grid(c.squeeze(), 
                    normalize=True, scale_each=True), global_step=0)
        
        x = init_x
        i = 0
        pbar = tqdm(__iter__)
        for step in pbar:
            ones_vec = torch.ones(self.sample_kwargs['batch_size'], device=self.device)
            if isinstance(step, float): 
                time_step = ones_vec * step # t,
            elif isinstance(step, Tuple):
                time_step = (ones_vec * step[0], ones_vec * step[1]) # (t, tminus1)
            else:
                raise NotImplementedError

            x, x_mean = self.predictor(
                score=self.score,
                sde=self.sde,
                x=x,
                time_step=time_step,
                step_size=step_size,
                cond_inp=c, 
                **self.sample_kwargs['predictor']
                )

            if logging:
                if i % logg_kwargs['num_img_in_log'] == 0:
                    writer.add_image('reco', torchvision.utils.make_grid(x_mean.squeeze(), normalize=True, scale_each=True), i)
                
                
                pbar.set_postfix({'step': step})
            i += 1

        if logging:
            writer.add_image(
                'final_reco', torchvision.utils.make_grid(x_mean.squeeze(),
                normalize=True, scale_each=True), global_step=0)

        return x_mean 