#Before running this script, please change the sys.path and yaml file path
import sys
sys.path.append('/home/haotian')
import os
import random
import argparse
import pickle
import yaml
import functools
from easydict import EasyDict
import numpy as np
from tqdm.auto import tqdm

import torch
from torch_geometric.data import Batch ,DataLoader
from torch_scatter import scatter_add
from SDE_CG_C import model,runner
from SDE_CG_C.utils.dataset import *
from SDE_CG_C.utils.transforms import *
from SDE_CG_C.utils import get_scheduler
from SDE_CG_C.model import marginal_prob_std
from SDE_CG_C.model import diffusion_coeff

with open('/home/haotian/SDE_CG_file/qm9_default.yml','rb') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

device = config.device
#move to the sirectory where SDE_CG locates.
task_directory = config.task_directory
val_pkl = os.path.join(task_directory,'data','val_QM9.pkl')
test_pkl = os.path.join(task_directory,'data','test_QM9.pkl')
train_pkl = os.path.join(task_directory,'data','train_QM9.pkl')
checkpoint = os.path.join(task_directory,'checkpoint')


# set random seed
np.random.seed(config.train.seed)
random.seed(config.train.seed)
torch.manual_seed(config.train.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
torch.backends.cudnn.benchmark = True
print('set seed for random, numpy and torch')


print('Loading dataset...')
tf = get_standard_transforms(order=3)
train_dset = MoleculeDataset(train_pkl, transform=tf)
val_dset = MoleculeDataset(val_pkl, transform=tf)
test_dset = MoleculeDataset(test_pkl, transform=tf)

sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = model.ScoreNet(config,marginal_prob_std_fn,device=device)
score_model = score_model.to(device)
loss_fn = model.loss_fn
optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, score_model.parameters()),
                        lr=config.train.optimizer.lr,
                        weight_decay=config.train.optimizer.weight_decay)
scheduler = get_scheduler(config, optimizer)

solver = runner.Runner(train_dset, val_dset, val_dset, score_model, marginal_prob_std_fn, optimizer, scheduler,  config, loss_fn, device, checkpoint)
solver.train()