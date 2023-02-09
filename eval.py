###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Kai Li
# LastEditTime: 2021-09-05 22:34:03
###

import re
from typing import OrderedDict
from nichang.utils import tensors_to_device
from nichang.videomodels import VideoModel, update_frcnn_parameter
from nichang.models.ctcnet import CTCNet
from nichang.datas.avspeech_dataset import AVSpeechDataset
from nichang.losses import PITLossWrapper, pairwise_neg_sisdr
from nichang.metrics import ALLMetricsTracker
import os
import os.path as osp
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")

# from nichang.models.avfrcnn_videofrcnn import AVFRCNNVideoFRCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--test_dir", type=str, required=True,
    help="Test directory including the json files"
)
parser.add_argument(
    "-c", "--conf_dir", default="local/lrs2_conf.yml",
    help="Full path to save best validation model"
)
parser.add_argument(
    "-s", "--save_dir", default=None,
    help="Full path to save the results wav"
)
parser.add_argument("--exp_dir", default="exp/tmp",
                    help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=-1,
    help="Number of audio examples to save, -1 means all"
)


compute_metrics = ["si_sdr", "sdr"]


def load_ckpt(path, submodule=None):
    _state_dict = torch.load(path, map_location="cpu")['state_dict']
    if submodule is None:
        return _state_dict

    state_dict = OrderedDict()
    for k, v in _state_dict.items():
        if submodule in k:
            L = len(submodule)
            state_dict[k[L+1:]] = v
    return state_dict


def main(conf):
    conf["exp_dir"] = os.path.join(
        "exp", conf["log"]["exp_name"])
    conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    print(model_path)
    sample_rate = conf["data"]["sample_rate"]
    audiomodel = CTCNet.from_pretrain(model_path, sample_rate=sample_rate, **conf["audionet"])
    videomodel = VideoModel(**conf["videonet"])

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    test_set = AVSpeechDataset(
        conf["test_dir"],
        n_src=conf["data"]["nondefault_nsrc"],
        sample_rate=conf["data"]["sample_rate"],
        segment=None,
        normalize_audio=conf["data"]["normalize_audio"],
        return_src_path=True
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf["exp_dir"], "results/")
    os.makedirs(ex_save_dir, exist_ok=True)
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    metrics = ALLMetricsTracker(
        save_file=os.path.join(ex_save_dir, "metrics.csv"))
    torch.no_grad().__enter__()

    pbar = tqdm(range(len(test_set)))
    for idx in pbar:
        # Forward the network on the mixture.
        mix, sources, target_mouths, key, src_path = tensors_to_device(
            test_set[idx], device=model_device)
        mouth_emb = videomodel(target_mouths.unsqueeze(0).float())
        est_sources = audiomodel(mix[None, None], mouth_emb)
        loss, reordered_sources = loss_func(
            est_sources, sources[None, None], return_ests=True)
        mix_np = mix
        sources_np = sources[None]
        est_sources_np = reordered_sources.squeeze(0)
        metrics(mix=mix_np, clean=sources_np, estimate=est_sources_np, key=key)
        if not (idx % 10):
            pbar.set_postfix(metrics.get_mean())

    metrics.final()
    mean, std = metrics.get_mean(), metrics.get_std()
    keys = list(mean.keys() & std.keys())
    
    order = ["sdr_i", "si-snr_i", "pesq", "stoi", "sdr", "si-snr"]
    def get_order(k):
        try:
            ind = order.index(k)
            return ind
        except ValueError:
            return 100

    keys.sort(key=get_order)
    for k in keys:
        m, s = mean[k], std[k]
        print(f"{k}\tmean: {m:.4f}  std: {s:.4f}")


if __name__ == "__main__":
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic['main_args'])
    main(def_conf)
