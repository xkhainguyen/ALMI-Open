import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
import torch

import wandb 

def train(args):
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # # print("args",args)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    
    
    # print("-----",env.args)
    #1/0
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    os.makedirs(ppo_runner.log_dir, exist_ok=True)
<<<<<<< HEAD
    # wandb.init(
    #             project='ALMI',
    #             name=args.task + '_' + train_cfg.runner.run_name, 
    #             config={
    #                     "env": class_to_dict(env_cfg),
    #                     "train": class_to_dict(train_cfg)
    #                 },
    #             dir=ppo_runner.log_dir,
    #             sync_tensorboard=True)
=======
    wandb.init(
                project='ALMI',
                name=args.task + '_' + train_cfg.runner.run_name, 
                config={
                        "env": class_to_dict(env_cfg),
                        "train": class_to_dict(train_cfg)
                    },
                dir=ppo_runner.log_dir,
                sync_tensorboard=True)
>>>>>>> 48278cfe2af9586269563fe95574e4fb4a9d3eeb
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
