import numpy as np
from src.utils.bins import Binner, BinSpec
from src.eval.metrics import classification_metrics, window_hit, plcc_srocc

def test_bins():
    b = Binner(BinSpec(1.0,5.0,0.5))
    assert b.num_classes() == 8
    assert b.to_class(3.1) in range(8)

def test_metrics():
    y_true = [0,1,2]; y_pred=[0,1,1]
    m = classification_metrics(y_true,y_pred)
    assert "acc" in m and "f1_macro" in m
    assert 0 <= m["acc"] <= 1
    wh = window_hit([3.0,2.0],[3.1,2.8],[0.5,0.5])
    assert 0 <= wh <= 1
    reg = plcc_srocc([1,2,3],[1.1,2.1,3.0])
    assert "plcc" in reg and "srocc" in reg
