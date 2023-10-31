from typing import Optional, Any, Dict, Tuple
import os 
import torch 
import torchvision
import numpy as np 
import functools 

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from .losses import score_based_loss_fn, epsilon_based_loss_fn
from .ema import ExponentialMovingAverage
from .sde import SDE, _SCORE_PRED_CLASSES, _EPSILON_PRED_CLASSES

from ..third_party_models import OpenAiUNetModel
from ..samplers import BaseSampler, wrapper_ddim
from ..ktc_methods import FastScoringFunction

def score_model_simple_trainer(
	score: OpenAiUNetModel,
	sde: SDE, 
	train_dl: DataLoader, 
	optim_kwargs: Dict,
	val_kwargs: Dict,
	device: Optional[Any] = None, 
	log_dir: str ='./',
	validation_data = None,
	clip_grad_norm = True  
	) -> None:

	writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
	optimizer = Adam(score.parameters(), lr=optim_kwargs['lr'])
	if any([isinstance(sde, classname) for classname in _SCORE_PRED_CLASSES]):
		loss_fn = score_based_loss_fn
	elif any([isinstance(sde, classname) for classname in _EPSILON_PRED_CLASSES]):
		loss_fn = epsilon_based_loss_fn
	else:
		raise NotImplementedError

	for epoch in range(optim_kwargs['epochs']):
		avg_loss, num_items = 0, 0
		score.train()
		c_test = None 
		x_test = None 
		for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
			c, x, _ = batch
			c = c.to(device)
			x = x.to(device)
			if idx == 0 and validation_data == None:
				c_test = c
				x_test = x

			loss = loss_fn(x, model=score, sde=sde, cond_inp=c)
			optimizer.zero_grad()
			loss.backward()

			if clip_grad_norm:
				torch.nn.utils.clip_grad_norm_(score.parameters(), 1.0)

			optimizer.step()

			avg_loss += loss.item() * x.shape[0]
			num_items += x.shape[0]
			if idx % optim_kwargs['log_freq'] == 0:
				writer.add_scalar('train/loss', loss.item(), epoch*len(train_dl) + idx) 
			if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
				ema = ExponentialMovingAverage(score.parameters(), decay=optim_kwargs['ema_decay'])
			if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
				ema.update(score.parameters())

		if (epoch % optim_kwargs['save_model_every_n_epoch']) == 0 or (epoch == optim_kwargs['epochs']-1):
			if epoch == optim_kwargs['epochs']-1:
				torch.save(score.state_dict(), os.path.join(log_dir, 'model.pt'))
				torch.save(ema.state_dict(), os.path.join(log_dir, 'ema_model.pt'))
			else:
				torch.save(score.state_dict(), os.path.join(log_dir, f'model_training.pt'))
				torch.save(ema.state_dict(), os.path.join(log_dir, f'ema_model_training.pt'))
			
		print('Average Loss: {:5f}'.format(avg_loss / num_items))
		writer.add_scalar('train/mean_loss_per_epoch', avg_loss / num_items, epoch + 1)
		if val_kwargs['sample_freq'] > 0:
			if epoch % val_kwargs['sample_freq']== 0:
				score.eval()
				
				c_test = validation_data[0].to(device)
				x_test = validation_data[1].to(device)

				sampler = BaseSampler(
					score=score,
					sde=sde,
					predictor=wrapper_ddim, 
					corrector=None,
					init_chain_fn=None,
					sample_kwargs={
						'num_steps': val_kwargs['num_steps'],
						'start_time_step': 0,
						'batch_size': c_test.shape[0],
						'im_shape': x.shape[1:],
						'eps': val_kwargs['eps'],
						'travel_length': 1, 
						'travel_repeat': 1, 
						'predictor': {'eta' : 0.9}
						},
					device=device)


				x_mean = sampler.sample(c_test, logging=False)
			
				x_round = torch.round(x_mean)
				x_round[x_round > 2] = 2.

				mean_score = 0
				for i in range(x_round.shape[0]):
					challenge_score = FastScoringFunction(x_test[i, 0, :, :].cpu().numpy(),
														 x_round[i, 0,:,:].cpu().numpy())
					mean_score += challenge_score 

				writer.add_scalar('val/challenge_score', mean_score/4, epoch) 

				sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
				writer.add_image('samples', sample_grid, global_step=epoch)
				
				gt_grid = torchvision.utils.make_grid(x_test, normalize=True, scale_each=True)
				writer.add_image('ground truth', gt_grid, global_step=epoch)



	# always save last model 
	torch.save(score.state_dict(), os.path.join(log_dir, 'model.pt'))
	torch.save(ema.state_dict(), os.path.join(log_dir, 'ema_model.pt'))