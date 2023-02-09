###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-05 18:12:18
###
import csv
import torch
import numpy as np
import logging
from pypesq import pesq
from pystoi import stoi

from ..losses import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr, singlesrc_neg_sisdr

logger = logging.getLogger(__name__)


class ALLMetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sdrs_i = []
        self.all_sisnrs = []
        self.all_sisnrs_i = []
        self.all_pesqs = []
        self.all_stois = []

        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i", "pesq", "stoi"]
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        self.pit_snr = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx")
        self.pit_sisnr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    def __call__(self, mix, clean, estimate, key):
        # sisnr
        sisnr = self.pit_sisnr(estimate.unsqueeze(0), clean.unsqueeze(0))
        mix = torch.stack([mix] * clean.shape[0], dim=0)
        sisnr_baseline = self.pit_sisnr(mix.unsqueeze(0), clean.unsqueeze(0))
        sisnr_i = sisnr - sisnr_baseline

        # sdr
        sdr = self.pit_snr(
            estimate.unsqueeze(0),
            clean.unsqueeze(0),
        )
        sdr_baseline = self.pit_snr(
            mix.unsqueeze(0),
            clean.unsqueeze(0),
        )
        sdr_i = sdr - sdr_baseline

        # stoi pesq
        # PESQ 
        _pesq = pesq(estimate.squeeze(0).cpu().numpy(), clean.squeeze(0).cpu().numpy(), 16000)

        # STOI
        _stoi = stoi(clean.squeeze(0).cpu().numpy(), estimate.squeeze(0).cpu().numpy(), 16000, extended=False)

        row = {
            "snt_id": key,
            "sdr": sdr.item(),
            "sdr_i": sdr_i.item(),
            "si-snr": -sisnr.item(),
            "si-snr_i": -sisnr_i.item(),
            "pesq": _pesq,
            "stoi": _stoi,
        }
        self.key = key
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(-sdr.item())
        self.all_sdrs_i.append(-sdr_i.item())
        self.all_sisnrs.append(-sisnr.item())
        self.all_sisnrs_i.append(-sisnr_i.item())
        self.all_pesqs.append(_pesq)
        self.all_stois.append(_stoi)

    def get_mean(self):
        return {
            "sdr": np.mean(self.all_sdrs),
            "sdr_i": np.mean(self.all_sdrs_i),
            "si-snr": np.mean(self.all_sisnrs),
            "si-snr_i": np.mean(self.all_sisnrs_i),
            "pesq": np.mean(self.all_pesqs),
            "stoi": np.mean(self.all_stois)
        }

    def get_std(self):
        return {
            "sdr": np.std(self.all_sdrs),
            "sdr_i": np.std(self.all_sdrs_i),
            "si-snr": np.std(self.all_sisnrs),
            "si-snr_i": np.std(self.all_sisnrs_i),
            "pesq": np.std(self.all_pesqs),
            "stoi": np.std(self.all_stois)
        }

    def final(
        self,
    ):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "sdr_i": np.array(self.all_sdrs_i).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "si-snr_i": np.array(self.all_sisnrs_i).mean(),
            "pesq": np.array(self.all_pesqs).mean(),
            "stoi": np.array(self.all_stois).mean()
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "sdr_i": np.array(self.all_sdrs_i).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "si-snr_i": np.array(self.all_sisnrs_i).std(),
            "pesq": np.array(self.all_pesqs).std(),
            "stoi": np.array(self.all_stois).std()
        }
        self.writer.writerow(row)
        self.results_csv.close()
    
