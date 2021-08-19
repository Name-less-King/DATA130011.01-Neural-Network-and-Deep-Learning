#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math

class SLBI(Optimizer):

	def __init__ (self, params, lr=required, kappa=1, mu=100, betas=(0.9,0.999), eps=1e-08, weight_decay=0, dampening=0):
		defaults = dict(lr=lr, kappa=kappa, mu=mu, betas=betas, eps=eps, weight_decay=weight_decay, dampening=dampening)
		print('*******************************************')
		for key in defaults:
			print(key, ' : ', defaults[key])
		print('*******************************************')
		super(SLBI, self).__init__(params, defaults)


	def __setstate__(self, state):
		super(SLBI, self).__setstate__(state)


	def assign_name(self, name_list):
		for group in self.param_groups:
			for iter, p in enumerate(group['params']):
				param_state = self.state[p]
				param_state['name'] = name_list[iter]


	def initialize_slbi(self, layer_list=None):
		if layer_list == None:
			pass
		else:
			for group in self.param_groups:
				for p in group['params']:
					param_state = self.state[p]
					if param_state['name'] in layer_list:
						param_state['z_buffer'] = torch.zeros_like(p.data)
						param_state['gamma_buffer'] = torch.zeros_like(p.data)


	def step(self, closure=None):
		loss = None
		if closure is not None:
			loss = closure()
		for group in self.param_groups:
			mu = group['mu']
			kappa = group['kappa']
			lr_kappa = group['lr'] * group['kappa']
			lr_gamma = group['lr'] / mu
			weight_decay = group['weight_decay']
			beta1, beta2 = group['betas']
			eps = group['eps']
			dampening = group['dampening']

			for p in group['params']:
				if p.grad is None:
					continue
				d_p = p.grad.data
				param_state = self.state[p]

				if 'step' not in param_state:
					param_state['step'] = 0
					param_state['exp_avg'] = torch.zeros_like(p)
					param_state['exp_avg_sq'] = torch.zeros_like(p)

				param_state['step'] += 1

				step = param_state['step']
				exp_avg = param_state['exp_avg']
				exp_avg_sq = param_state['exp_avg_sq']

				bias_correction1 = 1 - beta1 ** step
				bias_correction2 = 1 - beta2 ** step

				exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)
				denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

				step_size = lr_kappa / bias_correction1
				
				p.data.addcdiv_(exp_avg, denom, value=-step_size)

				if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
					d_p.add_(weight_decay, p.data)

				if  'z_buffer' in param_state:
					new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu 
					last_p = copy.deepcopy(p.data)				
					p.data.add_(-new_grad)
					param_state['z_buffer'].add_(-lr_gamma, param_state['gamma_buffer'] - last_p)
					if len(p.data.size()) == 2:
						param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
					elif len(p.data.size()) == 4:
						param_state['gamma_buffer'] = kappa * self.shrink_group(param_state['z_buffer'])
					else:
						pass
				else:
					p.data.add_(-lr_kappa, d_p)


	def calculate_w_star_by_layer(self, layer_name):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if  'z_buffer' in param_state and param_state['name'] == layer_name:
					if len(p.data.size()) == 2:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					elif len(p.data.size()) == 4:
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass
				else:
					pass



	def calculate_all_w_star(self):
		for group in self.param_groups:
			for p in group['params']:
				param_state = self.state[p]
				if  'z_buffer' in param_state:
					if len(p.data.size()) == 2:
	#					print(p.data.size())
	#					print(param_state['gamma_buffer'].size())
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					elif len(p.data.size()) == 4:
	#					print(p.data.size())
	#					print(param_state['gamma_buffer'].size())
						param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
					else:
						pass

		

	def calculate_layer_residue(self, layer_name):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if param_state['name'] == layer_name:
					if 'gamma_buffer' in param_state:
						diff = ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
					else:
						pass
		diff /= (2*mu)
		print('Residue of' + layer_name + ' : ', diff)


	def calculate_all_residue(self):
		diff = 0
		for group in self.param_groups:
			mu = group['mu']
			for p in group['params']:
				param_state = self.state[p]
				if 'gamma_buffer' in param_state:
					diff += ((p.data - param_state['gamma_buffer']) * (p.data - param_state['gamma_buffer'])).sum().item()
		diff /= (2*mu)
		print('Residue : ', diff)


	def shrink(self, s_t, lam):
		#proximal mapping for 2-d weight(fc layer)
		gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
		return gamma_t


	def shrink_group(self, ts):
		# shrinkage for 4-d weight(conv layer)
		ts_reshape = torch.reshape(ts,(ts.shape[0],-1))
		ts_norm = torch.norm(ts_reshape,2,1)
		ts_shrink = torch.max(torch.zeros_like(ts_norm),torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm),ts_norm))
		ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape,0,1),ts_shrink),0,1)
		ts_return = torch.reshape(ts_return,ts.shape)
		return ts_return
