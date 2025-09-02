import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"acc": acc, "f1_macro": f1m}

def window_hit(mos_true, mos_pred, sigma):
    mos_true = np.asarray(mos_true); mos_pred = np.asarray(mos_pred); sigma = np.asarray(sigma)
    hits = ((mos_pred >= mos_true - sigma) & (mos_pred <= mos_true + sigma)).astype(float)
    return hits.mean()

def plcc_srocc(y_true, y_pred):
    plcc = pearsonr(y_true, y_pred)[0]
    srocc = spearmanr(y_true, y_pred)[0]
    return {"plcc": plcc, "srocc": srocc}
