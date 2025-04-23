#!/usr/bin/env python
# coding: utf-8

# Standard libraries imports
import os

# Third party imports
import wandb
import argparse
import torch
from easydict import EasyDict as edict

from torch.utils.tensorboard import SummaryWriter

# Local application imports
from src.modules.defaults.wrappers import DefaultWrapper
from src.modules.defaults.trainer import Trainer

from src.modules.self_supervised.DINO.wrappers import DINOWrapper
from src.modules.self_supervised.DINO.trainer import DINOTrainer

# from src.modules.self_supervised.DINOv2.wrappers import DINOv2Wrapper
# from src.modules.self_supervised.DINOv2.trainer import DINOv2Trainer

from src.utils.system_def import *
from src.utils.launch import dist, launch, synchronize
from src.utils.helpfuns import load_params

global debug

def parse_arguments():
    parser = argparse.ArgumentParser(description="The main takes as argument the parameters dictionary from a json file")
    parser.add_argument("--params_path",type=str,required=False,default="./config/params.json",help="Give the path of the json file which contains the training parameters")
    parser.add_argument("--checkpoint", type=str, required=False, help="Give a valid checkpoint name")
    parser.add_argument("--test", action="store_true", default=False, help="Flag for testing")
    parser.add_argument("--find_lr", action="store_true", default=False, help="Flag for lr finder")
    parser.add_argument("--debug",action="store_true",default=False,help="Flag for turning on the debug_mode")
    parser.add_argument("--data_location", type=str, required=False, help="Update the datapath")
    parser.add_argument("--dist_url",type=str,default="env://",required=False,help="URL of master node, for use with SLURM")
    parser.add_argument( "--port",type=int,required=False,default=45124,help="Explicit port selection, for use with SLURM")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Self Supervised Learning arguments
    parser.add_argument("--dino", action="store_true", default=False, help="Flag for training with DINO")
    parser.add_argument("--dinov2", action="store_true", default=False, help="Flag for training with DINOv2")

    return parser.parse_args()


def update_params_from_args(params, args):
    if args.data_location:
        parameters["dataset_params"]["data_location"] = args.data_location

    # check self-supervised method
    assert not args.dino * args.dinov2, "DINO or DINOv2 can be on but not both"

 
def main(parameters, args):

    # define system
    define_system_params(parameters.system_params)

    # ======================= Instanciate WRAPPER ======================= #
    if args.dino or args.dinov2:
        if args.dino:
            wrapper = DINOWrapper(parameters)
        elif args.dinov2:
            #wrapper = DINOv2Wrapper(parameters, use_momentum=use_momentum)
            pass
    else:
        wrapper = DefaultWrapper(parameters)
    wrapper.instantiate()

    # ======================= Instanciate LOGGER ======================= #
    if wrapper.is_main_process:
        log_params = wrapper.parameters.log_params
        training_params = wrapper.parameters.training_params
        if wrapper.log_params["run_name"] == "DEFINED_BY_MODEL_NAME":
            log_params["run_name"] = training_params.model_name
        elif wrapper.log_params["run_name"] == "DEFINED_BY_PARAMS":
            if args.dino or args.dinov2:
                ssl_method = "dino" if args.dino else "dinov2"
            method = "TL" if not args.dino and not args.dinov2 else f"SSL-{ssl_method}"
            pretrained = "INet" if wrapper.parameters.model_params.pretrained else "rand"
            log_params["run_name"] = "DATASET-"+ wrapper.parameters.dataset_params.dataset + "_" + "METHOD-" + method + "_" + "BACKBONE-" + wrapper.parameters.model_params.backbone_type +  "-" + wrapper.parameters.model_params.transformers_params.pretrained_type + "-" + pretrained
            wrapper.parameters.training_params.model_name = "DATASET-"+ wrapper.parameters.dataset_params.dataset + "_" + "METHOD-" + method + "_" + "BACKBONE-" + wrapper.parameters.model_params.backbone_type +  "-" + wrapper.parameters.model_params.transformers_params.pretrained_type + "-" + pretrained
        if args.debug:
            os.environ["WANDB_MODE"] = "dryrun"
        if not (args.test or args.find_lr):
            print("Using WANDB logging")
            wandb.init(
                project=log_params.project_name,
                name=log_params.run_name,
                config=wrapper.parameters,
                resume=True if training_params.restore_session else False,
            )

    # define trainer
    if args.dino or args.dinov2:
        if args.dino:
            trainer = DINOTrainer(wrapper)
        elif args.dinov2:
            #trainer = DINOv2Trainer(wrapper, use_momentum=use_momentum)
            pass
    else:
        trainer = Trainer(wrapper)

    if args.test:
        trainer.test()
    elif args.find_lr:
        trainer.lr_grid_search(**wrapper.parameters.lr_finder.grid_search_params)
    else:
        trainer.train()
        if wrapper.is_supervised:
            trainer.test()


if __name__ == "__main__":
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)

    try:  
        launch(main, (parameters, args))
    except Exception as e:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    finally:
        if dist.is_initialized():
            synchronize()
            dist.destroy_process_group()