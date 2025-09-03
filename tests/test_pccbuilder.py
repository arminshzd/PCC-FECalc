import sys
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import contextlib
import os
import pytest

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

import FECalc.PCCBuilder as pccb
from FECalc.PCCBuilder import PCCBuilder


def make_settings(tmp_path):
    settings = {
        "ref_PCC_dir": "ref.pdb",
        "origin": [0, 0, 0],
        "anchor1": [0, 0, 0],
        "anchor2": [1, 0, 0],
        "pymol_dir": "/usr/bin/pymol",
    }
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(settings))
    return settings_file


def test_init_creates_directory_and_raises_on_file(tmp_path):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("AA", base_dir, settings_file)
    assert builder.PCC_dir.is_dir()

    file_base = tmp_path / "not_dir"
    file_base.write_text("content")
    with pytest.raises(ValueError):
        PCCBuilder("AA", file_base, settings_file)


def test_check_and_set_done(tmp_path):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("AA", base_dir, settings_file)

    stage_rel = "stage"
    stage_abs = builder.base_dir / "stage_abs"
    assert not builder._check_done(stage_rel)
    assert not builder._check_done(stage_abs)
    builder._set_done(stage_rel)
    builder._set_done(stage_abs)
    assert builder._check_done(stage_rel)
    assert builder._check_done(stage_abs)


def test_create_pcc_runs_pymol_and_marks_done(tmp_path, monkeypatch):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("AC", base_dir, settings_file)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr(pccb, "subprocess", SimpleNamespace(run=fake_run))

    def fake_read_pdb(path):
        if str(path).endswith("_babel.pdb"):
            coords = np.arange(15, dtype=float).reshape(5, 3)
        else:
            coords = np.arange(9, dtype=float).reshape(3, 3)
        return [], [], coords

    written = {}

    def fake_write(infile, outfile, coords):
        written["coords"] = np.array(coords)

    @contextlib.contextmanager
    def dummy_cd(path):
        current = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(current)

    monkeypatch.setattr(pccb, "_read_pdb", fake_read_pdb)
    monkeypatch.setattr(pccb, "_write_coords_to_pdb", fake_write)
    monkeypatch.setattr(pccb, "cd", dummy_cd)

    builder._create_pcc()

    assert any("PCCmold.py" in str(c) for c in calls)
    assert any("sub_preopt.sh" in str(c) for c in calls)
    assert written["coords"].shape == (3, 3)
    assert (builder.PCC_dir / ".done").exists()


def test_get_params_prep_pdb_called_and_log_checked(tmp_path, monkeypatch):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("DR", base_dir, settings_file)

    # create log file
    acpype_dir = builder.PCC_dir / "PCC.acpype"
    acpype_dir.mkdir()
    (acpype_dir / "acpype.log").write_text("all good\n")

    prep_args = []

    def fake_prep(infile, outfile, resname):
        prep_args.append((infile, outfile, resname))

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

    @contextlib.contextmanager
    def dummy_cd(path):
        current = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(current)

    monkeypatch.setattr(pccb, "_prep_pdb", fake_prep)
    monkeypatch.setattr(pccb, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(pccb, "cd", dummy_cd)

    builder._get_params()

    assert prep_args == [(f"{builder.PCC_code}_opt.pdb", f"{builder.PCC_code}_acpype.pdb", "PCC")]
    assert any("acpype" in str(c) for c in calls)
    assert (acpype_dir / ".done").exists()


def test_get_params_raises_on_warning(tmp_path, monkeypatch):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("DR", base_dir, settings_file)

    acpype_dir = builder.PCC_dir / "PCC.acpype"
    acpype_dir.mkdir()
    (acpype_dir / "acpype.log").write_text("Warning: bad\n")

    monkeypatch.setattr(pccb, "_prep_pdb", lambda *a, **k: None)
    monkeypatch.setattr(pccb, "subprocess", SimpleNamespace(run=lambda *a, **k: None))
    @contextlib.contextmanager
    def dummy_cd(path):
        current = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(current)

    monkeypatch.setattr(pccb, "cd", dummy_cd)

    with pytest.raises(RuntimeError):
        builder._get_params()


def test_minimize_pcc_runs_and_marks_done(tmp_path, monkeypatch):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("AA", base_dir, settings_file)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

    @contextlib.contextmanager
    def dummy_cd(path):
        current = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(current)

    monkeypatch.setattr(pccb, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(pccb, "cd", dummy_cd)

    builder._minimize_PCC(wait=False)

    assert any("sub_mdrun_em.sh" in str(c) for c in calls)
    assert (builder.PCC_dir / "em" / ".done").exists()


def test_create_orchestrates_and_skips_done(tmp_path, monkeypatch):
    settings_file = make_settings(tmp_path)
    base_dir = tmp_path / "calc"
    builder = PCCBuilder("AA", base_dir, settings_file)

    call_order = []

    monkeypatch.setattr(builder, "_create_pcc", lambda: call_order.append("create"))
    monkeypatch.setattr(builder, "_get_params", lambda: call_order.append("params"))
    monkeypatch.setattr(builder, "_minimize_PCC", lambda: call_order.append("min"))

    def check(stage):
        if stage == builder.PCC_dir:
            return False
        return True

    monkeypatch.setattr(builder, "_check_done", check)

    builder.create()
    assert call_order == ["create"]

    call_order.clear()
    monkeypatch.setattr(builder, "_check_done", lambda stage: True)
    builder.create()
    assert call_order == []
