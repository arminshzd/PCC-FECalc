import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

# ensure package root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc import postprocess
from FECalc.postprocess import (
    _block_anal_3d,
    _calc_region_int,
    _calc_deltaF,
)


def test_block_anal_3d_bins_and_folds():
    x = np.array([1, 2, 1, 2, 1, 2, 1, 2])
    y = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    z = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    weights = np.ones(8)
    res = _block_anal_3d(x, y, z, weights, KbT=2.5, nbins=[2, 2, 2], folds=2)
    assert res.shape[0] == 8
    assert res['f_0'].isna().sum() == 4
    assert res['f_1'].isna().sum() == 4
    low = res.loc[res.x == 1.25, 'f_0'].dropna().iloc[0]
    high = res.loc[res.x == 1.75, 'f_0'].dropna().iloc[0]
    assert np.isclose(low, -high, atol=1e-6)


def test_calc_region_int_and_deltaF():
    KbT = 2.5
    x_vals = [1, 1, 1, 1, 2, 2, 2, 2]
    y_vals = [0, 0, 1, 1, 0, 0, 1, 1]
    x = np.repeat(x_vals, 3)
    y = np.repeat(y_vals, 3)
    z = np.tile([0, 0.5, 1], 8)
    bound = pd.DataFrame({'x': x, 'y': y, 'z': z, 'F': np.zeros_like(x, dtype=float)})
    unbound = pd.DataFrame({'x': x, 'y': y, 'z': z, 'F': np.full_like(x, 0.1, dtype=float)})
    assert abs(_calc_region_int(bound.copy(), KbT)) < 2e-3
    assert _calc_region_int(unbound.copy(), KbT) == pytest.approx(0.1, abs=2e-3)
    assert _calc_deltaF(bound.copy(), unbound.copy(), KbT) == pytest.approx(-0.1, abs=1e-3)


def test_calc_FE_success(monkeypatch, tmp_path):
    colvars = pd.DataFrame({
        'time': [0, 1000, 2000, 3000],
        'pb.bias': [0, 0, 0, 0],
        'dcom': [0, 0, 0, 0],
        'ang': [0, 0, 0, 0],
        'v3cos': [0, 0, 0, 0],
        'weights': [1, 1, 1, 1],
    })
    block_df = pd.DataFrame({
        'x': [1.0, 3.0],
        'y': [0, 0],
        'z': [0, 0],
        'f_0': [0.0, 0.0],
        'f_1': [0.0, 0.0],
        'ste': [0.1, 0.1],
    })
    monkeypatch.setattr(postprocess, '_load_plumed', lambda p, KbT: colvars)
    monkeypatch.setattr(postprocess, '_block_anal_3d', lambda *a, **k: block_df)
    monkeypatch.setattr(postprocess, '_get_box_size', lambda p: 10.0)
    calls = iter([5.0, 5.0])
    monkeypatch.setattr(postprocess, '_calc_deltaF', lambda *args, **kwargs: next(calls))
    fe, se = postprocess._calc_FE(tmp_path, 2.5, 0, 2)
    assert fe == 5.0
    assert se == 0.0


def test_calc_FE_raises_when_all_blocks_discarded(monkeypatch, tmp_path):
    colvars = pd.DataFrame({
        'time': [0, 1000],
        'pb.bias': [0, 0],
        'dcom': [0, 0],
        'ang': [0, 0],
        'v3cos': [0, 0],
        'weights': [1, 1],
    })
    block_df = pd.DataFrame({
        'x': [1.0],
        'y': [0],
        'z': [0],
        'f_0': [0.0],
        'ste': [0.1],
    })
    monkeypatch.setattr(postprocess, '_load_plumed', lambda p, KbT: colvars)
    monkeypatch.setattr(postprocess, '_block_anal_3d', lambda *a, **k: block_df)
    monkeypatch.setattr(postprocess, '_get_box_size', lambda p: 10.0)
    def _raise(*args, **kwargs):
        raise ValueError('fail')
    monkeypatch.setattr(postprocess, '_calc_deltaF', _raise)
    with pytest.raises(ValueError):
        postprocess._calc_FE(tmp_path, 2.5, 0, 1)


def test_postprocess_wrapper_writes_metadata(monkeypatch, tmp_path):
    (tmp_path / 'complex').mkdir()
    monkeypatch.setattr(postprocess, '_calc_FE', lambda i, K, t, n: (1.0, 0.1))
    monkeypatch.setattr(postprocess, '_calc_K', lambda fe, fe_err, KbT, box: (2.0, 0.2))
    postprocess.postprocess_wrapper('PCC', 'TARGET', tmp_path, 300, 0, 1, 1.0)
    data = json.loads((tmp_path / 'metadata.JSON').read_text())
    assert data['FE'] == 1.0
    assert data['K'] == 2.0
