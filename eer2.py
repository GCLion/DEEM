import numpy as np
import torch
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import sys
import numpy as np
from torch.utils.data import DataLoader

import eval_metrics as em


def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def calculate_min_tDCF(y_true, y_score):
    Cfa = 1
    Cfr = 10
    Ptar = 0.05
    Pnon = 1 - Ptar
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    DCF = Cfa * fpr * Pnon + Cfr * (1 - tpr) * Ptar
    min_DCF = np.min(DCF)
    min_index = np.argmin(DCF)
    thresh = thresholds[min_index]
    return min_DCF, thresh


def getmetric(cm_scores):
    cm_scores = np.array(cm_scores)
    cm_1 = np.genfromtxt(
        "/data0/project/zhoutreeman/2021/data/LA/ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", dtype=str)
    cm_keys = cm_1[:, 4]
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    asv_score_file = "/data0/project/zhoutreeman/2021/data/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)
    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                                                     asv_threshold)
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }
    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF_legacy(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
                                                       cost_model,
                                                       True)
    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    return eer_cm * 100, min_tDCF

def getmetricFor(cm_scores,Y):
    cm_scores = np.array(cm_scores)
    cm_keys = np.array(Y)
    bona_cm = cm_scores[cm_keys == 1]
    spoof_cm = cm_scores[cm_keys == 0]
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    return eer_cm * 100

def getmetricwild(cm_scores):
    cm_scores = np.array(cm_scores)
    cm_1 = np.genfromtxt(
        "meta.csv",delimiter=',', dtype=str)
    cm_keys = cm_1[:, 2]
    bona_cm = cm_scores[cm_keys == 'bona-fide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    return eer_cm * 100


def getmetriclist(cm_scores):
    eerlist = dict()
    cm_scores = np.array(cm_scores)
    cm_1 = np.genfromtxt(
        "ASVspoof2019.LA.cm.eval.trl.txt", dtype=str)
    keys = cm_1[:, 4]
    bona_cm = cm_scores[keys == 'bonafide']
    type = cm_1[:, 3]
    typeset = set(type)

    for t in typeset:
        if t == '-':
            continue
        spoof_cm = cm_scores[type == t]
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
        eerlist[t] = eer_cm * 100
    return eerlist


def produce_evaluation_file(dataset, model, device):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            score_list = []
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]
            ).data.cpu().numpy().ravel()
            # add outputs
            score_list.extend(batch_score.tolist())
        eer, min_tdcf = getmetric(score_list)
        return eer, min_tdcf


if __name__ == '__main__':
    cm_score_file = "20194.txt"
    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_1 = np.genfromtxt(
        "ASVspoof2019.LA.cm.eval.trl.txt", dtype=str)
    cm_scores = cm_data[:, 1].astype(np.float)
    cm_keys = cm_1[:, 4]
    y_true = []
    for i in cm_keys:
        if i == 'spoof':
            y_true.append(0)
        else:
            y_true.append(1)
    y_true = list(map(float, y_true))
    y_true = np.array(y_true)
    cm_scores = list(map(float, cm_scores))
    cm_scores = np.array(cm_scores)

    eer, thresh1 = calculate_eer(y_true, cm_scores)
    min_tDCF, thresh2 = calculate_min_tDCF(y_true, cm_scores)
    print("EER: {:.4f}".format(eer))
    print("Threshold: {:.4f}".format(thresh1))
    print("Min-tDCF: {:.4f}".format(min_tDCF))
    print("Threshold: {:.4f}".format(thresh2))
