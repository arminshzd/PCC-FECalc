import sys
from pathlib import Path
import json
import numpy as np
import pytest

# ensure package root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.postprocess import _load_plumed, _get_box_size, _calc_K, _write_report


def test_load_plumed_reads_fields(tmp_path):
    colvar_content = (
        "#! FIELDS time pb.bias dcom ang v3cos\n"
        "0 0.0 1 2 3\n"
        "1 1.0 4 5 6\n"
    )
    colvar_file = tmp_path / "COLVAR"
    colvar_file.write_text(colvar_content)

    data = _load_plumed(colvar_file, KbT=1000)
    assert list(data.columns) == ["time", "pb.bias", "dcom", "ang", "v3cos", "weights"]
    expected = np.exp(data["pb.bias"])  # since KbT=1000
    assert np.allclose(data["weights"], expected)


def test_get_box_size_returns_last_line(tmp_path):
    gro_content = (
        "test\n"
        "1\n"
        "1SOL H1 1 0 0 0\n"
        "2.5 2.5 2.5\n"
    )
    gro_file = tmp_path / "box.gro"
    gro_file.write_text(gro_content)
    assert _get_box_size(gro_file) == 2.5


def test_calc_K_computation():
    K, K_err = _calc_K(-5.0, 0.5, 2500.0, 1.0)
    A = 1/(6.022e23*1.0**3*1e-24)
    B = 1000/2500.0
    expected_K = A*np.exp(B*(-5.0))*1e6
    expected_err = B*expected_K*0.5
    assert np.isclose(K, expected_K)
    assert np.isclose(K_err, expected_err)


def test_write_report_creates_json(tmp_path):
    out = tmp_path / "report.json"
    _write_report("PCC", "TARGET", out, 1.0, 0.1, 2.0, 0.2)
    data = json.loads(out.read_text())
    assert data["PCC"] == "PCC"
    assert data["Target"] == "TARGET"
    assert data["FE"] == 1.0
    assert data["K_err"] == 0.2


def test_load_plumed_ignores_comments(tmp_path):
    colvar_content = (
        "#! FIELDS time pb.bias dcom ang v3cos\n"
        "0 0.0 1 2 3\n"
        "# comment line\n"
        "1 1.0 4 5 6\n"
    )
    colvar_file = tmp_path / "COLVAR"
    colvar_file.write_text(colvar_content)
    data = _load_plumed(colvar_file, KbT=1000)
    assert len(data) == 2


def test_get_box_size_raises_on_bad_last_line(tmp_path):
    gro_content = (
        "test\n"
        "1\n"
        "1SOL H1 1 0 0 0\n"
        "not-a-number\n"
    )
    gro_file = tmp_path / "box.gro"
    gro_file.write_text(gro_content)
    with pytest.raises(ValueError):
        _get_box_size(gro_file)
