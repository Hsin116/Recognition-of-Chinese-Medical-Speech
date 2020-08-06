#!/usr/bin/env python
# coding: utf-8
import yaml
import torch
import random
import argparse
import numpy as np
import os



# Make cudnn CTC deterministic
torch.backends.cudnn.deterministic = True

# Arguments
parser = argparse.ArgumentParser(description='Training E2E asr.')
parser.add_argument('--config',default="./config/example.yaml", type=str, help='Path to experiment config.')
parser.add_argument('--name', default=None, type=str, help='Name for logging.')
parser.add_argument('--logdir', default='log/try', type=str, help='Logging path.', required=False)
parser.add_argument('--ckpdir', default='try', type=str, help='Checkpoint/Result path.', required=False)
parser.add_argument('--load', default=None, type=str, help='Load pre-trained model', required=False)
parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
parser.add_argument('--njobs', default=1, type=int, help='Number of threads for decoding.', required=False)
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--test', action='store_true', help='Test the model.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')

# add by Leslie
parser.add_argument('--gpu-num', default=0, type=int, help='the number of gpu to work')
parser.add_argument('--spec-aug', dest='spec_aug', action='store_true', help='use specaugment')
parser.add_argument('--model', default='asr', type=str, help='Model.', required=False)
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--C2E', default='./config/', type=str, help='C2E.json path.', required=False)
parser.add_argument('--start-epoch', default=1, type=int, help='start epoch for test')
parser.add_argument('--end-epoch', default=100, type=int, help='end epoch for test')

paras = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(paras.gpu_num) # GPU

setattr(paras,'gpu',not paras.cpu)
setattr(paras,'verbose',not paras.no_msg)
config = yaml.load(open(paras.config,'r'))

random.seed(paras.seed)
np.random.seed(paras.seed)
torch.manual_seed(paras.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(paras.seed)


if not paras.test:
    # Train ASR
    from src.solver import Trainer as Solver

    solver = Solver(config, paras)
    solver.load_data()
    if paras.continue_from:
        solver.load_pre_model()
    else:
        solver.set_model()
    solver.exec()

else:
    # Test ASR
    from src.solver import Test_iter as Solver
    for i in range(paras.start_epoch, paras.end_epoch):
        paras.model = 'asr_' + str(i)
        solver = Solver(config, paras)
        solver.load_data()
        solver.set_model()
        solver.exec()
