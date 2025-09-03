import sys
import json
import subprocess
from pathlib import Path

import pytest

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.TargetMOL import TargetMOL


def _write_settings(tmp_path, **overrides):
    data = {
        "name": "MOL",
        "output_dir": str(tmp_path / "out"),
        "input_pdb_dir": str(tmp_path / "in.pdb"),
        "charge": 0,
        "anchor1": [0, 0, 0],
        "anchor2": [1, 1, 1],
    }
    data.update(overrides)
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(data))
    return settings_file


def _init_target(tmp_path):
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")
    settings = _write_settings(tmp_path)
    return TargetMOL(settings)


def test_init_validates_paths(tmp_path):
    pdb = tmp_path / "in.pdb"
    pdb.write_text("ATOM\n")

    # missing output_dir
    s1 = _write_settings(tmp_path)
    data = json.loads(s1.read_text())
    del data["output_dir"]
    s1.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        TargetMOL(s1)

    # output_dir exists as file
    out_file = tmp_path / "file.txt"
    out_file.write_text("")
    s2 = _write_settings(tmp_path, output_dir=str(out_file))
    with pytest.raises(ValueError):
        TargetMOL(s2)

    # missing input_pdb_dir
    s3 = _write_settings(tmp_path)
    data = json.loads(s3.read_text())
    del data["input_pdb_dir"]
    s3.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        TargetMOL(s3)

    # input_pdb_dir not a file
    missing = tmp_path / "missing.pdb"
    s4 = _write_settings(tmp_path, input_pdb_dir=str(missing))
    with pytest.raises(ValueError):
        TargetMOL(s4)


def test_check_and_set_done_handle_paths(tmp_path):
    tm = _init_target(tmp_path)

    # relative stage
    tm._set_done(Path("rel"))
    assert (tm.base_dir / "rel" / ".done").exists()
    assert tm._check_done(Path("rel"))

    # absolute stage outside base_dir
    abs_stage = tmp_path / "abs"
    tm._set_done(abs_stage)
    assert (abs_stage / ".done").exists()
    assert tm._check_done(abs_stage)


def test_get_params_runs_acpype(tmp_path, monkeypatch):
    tm = _init_target(tmp_path)

    acpype_dir = tm.base_dir / "MOL.acpype"
    acpype_dir.mkdir()
    (acpype_dir / "acpype.log").write_text("all good")

    prep_calls = []

    def fake_prep(infile, outfile, res):
        prep_calls.append((infile, outfile, res))
        Path(outfile).write_text("prepared")

    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        if isinstance(cmd, str) and cmd.startswith("cp"):
            _, src, dst = cmd.split()
            src_path = Path(src)
            dst_path = Path(dst)
            if dst == "." or dst_path.is_dir():
                (dst_path / src_path.name).write_text(src_path.read_text())
            else:
                dst_path.write_text(src_path.read_text())
        elif isinstance(cmd, list) and cmd[0] == "cp":
            src = Path(cmd[1])
            dst = Path(cmd[2])
            if cmd[2] == "." or dst.is_dir():
                (dst / src.name).write_text(src.read_text())
            else:
                dst.write_text(src.read_text())
        elif isinstance(cmd, str) and cmd.startswith("acpype"):
            pass
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("FECalc.TargetMOL._prep_pdb", fake_prep)
    monkeypatch.setattr(subprocess, "run", fake_run)

    tm._get_params()

    assert (tm.base_dir / "MOL.pdb").exists()
    assert prep_calls == [("MOL.pdb", "MOL_acpype.pdb", "MOL")]
    assert any(c.startswith("acpype") for c in commands)
    assert (acpype_dir / ".done").exists()


def test_get_params_raises_on_warning(tmp_path, monkeypatch):
    tm = _init_target(tmp_path)
    acpype_dir = tm.base_dir / "MOL.acpype"
    acpype_dir.mkdir()
    (acpype_dir / "acpype.log").write_text("Warning: issue")

    def fake_prep(infile, outfile, res):
        Path(outfile).write_text("prepared")

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, str) and cmd.startswith("cp"):
            _, src, dst = cmd.split()
            src_path = Path(src)
            dst_path = Path(dst)
            if dst == "." or dst_path.is_dir():
                (dst_path / src_path.name).write_text(src_path.read_text())
            else:
                dst_path.write_text(src_path.read_text())
        elif isinstance(cmd, list) and cmd[0] == "cp":
            src = Path(cmd[1])
            dst = Path(cmd[2])
            if cmd[2] == "." or dst.is_dir():
                (dst / src.name).write_text(src.read_text())
            else:
                dst.write_text(src.read_text())
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("FECalc.TargetMOL._prep_pdb", fake_prep)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        tm._get_params()


@pytest.mark.parametrize("wait_flag", [True, False])
def test_minimize_mol_copies_files_and_runs(tmp_path, monkeypatch, wait_flag):
    tm = _init_target(tmp_path)

    # set up acpype files
    acpype = tm.base_dir / "MOL.acpype"
    acpype.mkdir()
    for fname in ["MOL_GMX.gro", "MOL_GMX.itp", "posre_MOL.itp"]:
        (acpype / fname).write_text("test")

    commands = []

    def fake_run(cmd, *args, **kwargs):
        commands.append(cmd if isinstance(cmd, str) else " ".join(cmd))
        if isinstance(cmd, list) and cmd[0] == "cp":
            src = Path(cmd[1])
            dst = Path(cmd[2])
            if cmd[2] == "." or dst.is_dir():
                (dst / src.name).write_text(src.read_text())
            else:
                dst.write_text(src.read_text())
        elif isinstance(cmd, str) and cmd.startswith("cp"):
            _, src, dst = cmd.split()
            src_path = Path(src)
            dst_path = Path(dst)
            if dst == "." or dst_path.is_dir():
                (dst_path / src_path.name).write_text(src_path.read_text())
            else:
                dst_path.write_text(src_path.read_text())
        elif isinstance(cmd, str) and cmd.startswith("sed -i"):
            target = tm.base_dir / "em" / "topol.top"
            text = target.read_text().replace("PCC", "MOL")
            target.write_text(text)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    tm._minimize_MOL(wait=wait_flag)

    em_dir = tm.base_dir / "em"
    assert (em_dir / "MOL_GMX.gro").exists()
    assert "MOL" in (em_dir / "topol.top").read_text()
    sbatch_cmd = [c for c in commands if isinstance(c, str) and c.startswith("sbatch")][0]
    if wait_flag:
        assert "--wait" in sbatch_cmd
    else:
        assert "--wait" not in sbatch_cmd
    assert (em_dir / ".done").exists()


def test_export_moves_files(tmp_path, monkeypatch):
    tm = _init_target(tmp_path)

    em_dir = tm.base_dir / "em"
    em_dir.mkdir()
    (em_dir / "MOL_GMX.itp").write_text("itp")
    (em_dir / "posre_MOL.itp").write_text("posre")
    (em_dir / "MOL_em.pdb").write_text("pdb")

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd[0] == "cp":
            src = Path(cmd[1])
            dst = Path(cmd[2])
            if cmd[2] == "." or dst.is_dir():
                (dst / src.name).write_text(src.read_text())
            else:
                dst.write_text(src.read_text())
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    tm._export()

    export_dir = tm.base_dir / "export"
    assert (export_dir / "MOL.itp").read_text() == "itp"
    assert (export_dir / "posre_MOL.itp").read_text() == "posre"
    assert (export_dir / "MOL.pdb").read_text() == "pdb"
    assert (export_dir / ".done").exists()


def test_create_runs_stages_conditionally(tmp_path, monkeypatch):
    tm = _init_target(tmp_path)

    calls = []

    def fake_get_params(self):
        calls.append("get")
        self._set_done(self.base_dir / "MOL.acpype")

    def fake_minimize(self, wait=True):
        calls.append("min")
        self._set_done(self.base_dir / "em")

    def fake_export(self):
        calls.append("exp")
        self._set_done(self.base_dir / "export")

    monkeypatch.setattr(TargetMOL, "_get_params", fake_get_params)
    monkeypatch.setattr(TargetMOL, "_minimize_MOL", fake_minimize)
    monkeypatch.setattr(TargetMOL, "_export", fake_export)

    tm.create()
    assert calls == ["get", "min", "exp"]

    # now all stages done, should skip
    calls.clear()
    tm.create()
    assert calls == []

    # remove export done to test partial rerun
    (tm.base_dir / "export" / ".done").unlink()
    calls.clear()
    tm.create()
    assert calls == ["exp"]
