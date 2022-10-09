import os
import logging

import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rtpt import RTPT

import ttools

from marionet import datasets, models, callbacks
from marionet.interfaces import Interface
from scripts.train import main
import itertools



if __name__ == '__main__':
    parser = ttools.BasicArgumentParser()

    # Representation
    parser.add_argument("--layer_size", type=int, default=8,
                        help="size of anchor grid")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument("--num_classes", type=int, default=150,
                        help="size of dictionary")
    parser.add_argument("--canvas_size", type=int, default=128,
                        help="spatial size of the canvas")

    # Model
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_bg", type=str)
    parser.add_argument("--no_layernorm", action='store_true', default=False)

    # Training options
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=60000)
    parser.add_argument("--w_beta", type=float, default=0.002)
    parser.add_argument("--aow", type=float, default=0.0)
    parser.add_argument("--w_probs", type=float, nargs='+', default=[5e-3])
    parser.add_argument("--lr_bg", type=float, default=1e-3)
    parser.add_argument("--shuffle_all", action='store_true', default=False)
    parser.add_argument("--crop", action='store_true', default=False)
    parser.add_argument("--background", action='store_true', default=False)
    parser.add_argument("--sprites", action='store_true', default=False)
    parser.add_argument("--no_spatial_transformer", action='store_true',
                        default=False)
    parser.add_argument("--spatial_transformer_bg", action='store_true',
                        default=False)
    parser.add_argument("--straight_through_probs", action='store_true',
                        default=False)

    parser.set_defaults(num_worker_threads=4, bs=4, lr=1e-4, )

    args = parser.parse_args()
    argsDict = vars(args)

    experiment_sets = [
        {
            'seed': range(2, 3),
            'layer_size': np.array([16]),
            'num_layers': np.array([2]),
            'num_classes': np.array([54]),
            'aow': np.array([v for v in np.logspace(2, 3, 1)])
        },
    ]

    total_experiments = sum(len(list(itertools.product(*[v for v in experiment_set.values()])))
                            for experiment_set in experiment_sets)
    i = 0
    for experiment_set in experiment_sets:
        config_lists = [v for v in experiment_set.values()]
        configs = itertools.product(*config_lists)
        params = [k for k in experiment_set.keys()]

        for conf in configs:
            conf = [c.item() if isinstance(c, float) else c for c in conf]
            argsDict['checkpoint_dir'] = "out_tmp/SpaceInvaders_" \
                                         + "_".join([f'{key}_{c}' for c, key in zip(conf, experiment_set.keys())])
            for c, key in zip(conf, experiment_set.keys()):
                argsDict[key] = c
            print(args)
            print("=========" * 10)
            print("==========", "Starting experiment with the following cfg:", "==========",
                  f"[{i}/{total_experiments}]", "==========")
            print("=========" * 10)
            i += 1
            main(args)
