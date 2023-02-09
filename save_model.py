###
# Author: Kai Li
# Date: 2021-06-20 00:21:33
# LastEditors: Kai Li
# LastEditTime: 2021-09-09 23:12:28
###
# import comet_ml
import os
import argparse
import json
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from nichang.datas import AVSpeechDataset
from nichang.system.optimizers import make_optimizer
from nichang.system.core import System
# from nichang.system.comet import CometLogger
from nichang.system.tensorboard import TensorBoardLogger
from nichang.losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr
# from nichang.models.avfrcnn_transformer import AVFRCNNTransformer
from nichang.models.ctcnet import AVFRCNN2
from nichang.videomodels import FRCNNVideoModel, update_frcnn_parameter

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--exp_dir", default="exp/tmp", 
    help="Full path to save best validation model")
parser.add_argument(
    "-c", "--conf_dir", default="local/conf.yml", 
    help="Full path to save best validation model")
parser.add_argument(
    "-n", "--name", default=None, 
    help="Experiment name")
parser.add_argument(
    "--gpus", type=int, default=8,
    help="#gpus of each node")
parser.add_argument(
    "--nodes", type=int, default=1,
    help="#node")


# class TbAVSampler(Callback):
#     """Show samples of  
#         mixed wav, wav1 gt/est, wav2 gt/est;
#         video1, video2;
#     """
#     def __init__(self, n_samples=3):
#         self.n_samples = n_samples

#     def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         dataset = pl_module.val_loader.dataset
#         dataset.return_src_path = True
#         for i in range(self.n_samples):
#             idx = random.randint(0, len(dataset)-1)
#             inputs, targets, target_mouths, _ = dataset[idx]
#             est_targets = pl_module(inputs, target_mouths)

#         return super().on_epoch_start(trainer, pl_module)


def build_dataloaders(conf):
    train_set = AVSpeechDataset(
        conf["data"]["train_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=conf["data"]["segment"],
        normalize_audio=conf["data"]["normalize_audio"],
    )
    val_set = AVSpeechDataset(
        conf["data"]["valid_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        # segment=conf["data"]["segment"],
        segment=None,
        normalize_audio=conf["data"]["normalize_audio"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    return train_loader, val_loader



def main(conf):

    train_loader, val_loader = build_dataloaders(conf)

    # Define model and optimizer
    sample_rate = conf["data"]["sample_rate"]
    videomodel = FRCNNVideoModel(**conf["videonet"])
    audiomodel = AVFRCNN2(sample_rate=sample_rate, **conf["audionet"])
    optimizer = make_optimizer(audiomodel.parameters(), **conf["optim"])

    # for k, v in audiomodel.state_dict().items():
    #     print(k)

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)

    # Just after instantiating, save the args. Easy loading in the future.
    conf["main_args"]["exp_dir"] = os.path.join("exp", conf["log"]["exp_name"])
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = {
        # "train": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        "train": PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx"),
        "val": PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx"),
        }
    system = System(
        audio_model=audiomodel,
        video_model=videomodel,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    state_dict = torch.load("/data/home/scv1134/run/chenhang/nichang-avdomain/egs/frcnn2_para/exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/checkpoints/epoch=129-val_loss=-12.12.ckpt")
    for k, v in state_dict["state_dict"].items():
        print(k)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, "best_model.pth")


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    if args.name is not None:
        def_conf['log']['exp_name'] = args.name

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic)
    main(def_conf)


    # parser = prepare_parser_from_dict(def_conf, parser=parser)
    # arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    # main(arg_dic)
