###
# Author: Kai Li
# Date: 2022-04-03 08:50:42
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-03 18:02:56
###
###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-07 23:17:39
###

from typing import OrderedDict
from nichang.videomodels import VideoModel
from nichang.models.ctcnet import CTCNet
from nichang.datas.transform import get_preprocessing_pipelines
import os
import soundfile as sf
import torch
import yaml
import argparse
import numpy as np
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--conf_dir", default="exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/conf.yml",
    help="Full path to save best validation model"
)

def main(conf):
    conf["exp_dir"] = os.path.join(
        "exp", conf["log"]["exp_name"])
    conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "checkpoints/last.ckpt")
    model_path = "exp/vox2_10w_frcnn2_64_64_3_adamw_1e-1_blocks16_pretrain/best_model.pth"
    sample_rate = conf["data"]["sample_rate"]
    audiomodel = CTCNet(sample_rate=sample_rate, **conf["audionet"])
    ckpt = torch.load(model_path, map_location="cpu")['state_dict']
    audiomodel.load_state_dict(ckpt)
    videomodel = VideoModel(**conf["videonet"])

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    for idx in range(1, 2):
        spk, sr = sf.read("test_videos/interview/interview.wav", dtype="float32")
        mouth = get_preprocessing_pipelines()["val"](np.load("test_videos/interview/mouthroi/speaker{}.npz".format(idx))["data"])
        key = "spk{}".format(idx)
        
        # Forward the network on the mixture.
        target_mouths = torch.from_numpy(mouth).to(model_device)
        mix = torch.from_numpy(spk).to(model_device)
        # import pdb; pdb.set_trace()
        mouth_emb = videomodel(target_mouths.unsqueeze(0).unsqueeze(1).float())
        est_sources = audiomodel(mix[None, None], mouth_emb)

        gt_dir = "./test/sep_result"
        os.makedirs(gt_dir, exist_ok=True)
        # import pdb; pdb.set_trace()
        sf.write(os.path.join(gt_dir, key+".wav"), est_sources.squeeze(0).squeeze(0).cpu().numpy(), 16000)
        # import pdb; pdb.set_trace()


if __name__ == "__main__":
    from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()

    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    arg_dic = parse_args_as_dict(parser)
    def_conf.update(arg_dic['main_args'])
    main(def_conf)
