import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

# ensure package root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import FECalc.FECalc as fe_mod

from FECalc.FECalc import FECalc


def test_update_mdp_sets_temperature_and_steps(tmp_path):
    template = tmp_path / "in.mdp"
    template.write_text("ref_t = 100 100\nnsteps = 500\ngen_temp = 100\n")
    out = tmp_path / "out.mdp"

    fe = FECalc.__new__(FECalc)
    fe.T = 310

    fe.update_mdp(template, out, n_steps=1000)
    lines = out.read_text().splitlines()

    assert any("ref_t" in line and str(fe.T) in line for line in lines)
    assert any("gen_temp" in line and str(fe.T) in line for line in lines)
    assert any(line.strip().startswith("nsteps") and "1000" in line for line in lines)


def test_update_mdp_retains_steps_when_not_provided(tmp_path):
    template = tmp_path / "in.mdp"
    template.write_text("nsteps = 500\n")
    out = tmp_path / "out.mdp"

    fe = FECalc.__new__(FECalc)
    fe.T = 300

    fe.update_mdp(template, out)
    assert "nsteps = 500" in out.read_text()


def test_is_continuous_identifies_gaps():
    fe = FECalc.__new__(FECalc)
    assert fe._is_continuous([1, 2, 3]) is True
    assert fe._is_continuous([1, 3, 4]) is False


def _fake_run(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 0)


def test_create_plumed_replaces_placeholders(tmp_path):
    plumed_in = tmp_path / "plumed.dat"
    plumed_in.write_text("MOL ${2}\nPCC ${1}\nA ${3}\nB ${4}\n")
    plumed_out = tmp_path / "out.dat"

    fe = FECalc.__new__(FECalc)
    fe.MOL_list = [20, 21]
    fe.MOL_list_atom = ["M1", "M2"]
    fe.PCC_list = [10, 11, 12]
    fe.PCC_list_atom = ["O1", "A1", "A2"]
    fe.pcc = SimpleNamespace(origin=["O1"], anchor_point1=["A1"], anchor_point2=["A2"])
    fe.target = SimpleNamespace(anchor_point1=["M1"], anchor_point2=["M2"])
    fe.metad_bias_factor = 20
    fe.metad_pace = 100
    fe.T = 300
    fe.metad_height = 5

    fe._create_plumed(plumed_in, plumed_out)
    content = plumed_out.read_text()
    assert "${" not in content
    assert "20-21" in content
    assert "10-12" in content


def test_create_plumed_requires_continuous_ids(tmp_path):
    plumed_in = tmp_path / "plumed.dat"
    plumed_in.write_text("MOL ${2}\nPCC ${1}\n")
    plumed_out = tmp_path / "out.dat"

    fe = FECalc.__new__(FECalc)
    fe.MOL_list = [20, 22]
    fe.MOL_list_atom = ["M1", "M2"]
    fe.PCC_list = [10, 11]
    fe.PCC_list_atom = ["O1", "A1"]
    fe.pcc = SimpleNamespace(origin=["O1"], anchor_point1=["A1"], anchor_point2=["A1"])
    fe.target = SimpleNamespace(anchor_point1=["M1"], anchor_point2=["M2"])
    fe.metad_bias_factor = 20
    fe.metad_pace = 100
    fe.T = 300
    fe.metad_height = 5

    with pytest.raises(AssertionError):
        fe._create_plumed(plumed_in, plumed_out)


def test_mix_raises_when_complex_pdb_missing(tmp_path, monkeypatch):
    fe = FECalc.__new__(FECalc)
    fe.complex_dir = tmp_path / "complex"
    fe.complex_dir.mkdir()
    fe.target_dir = tmp_path / "target"
    fe.target_dir.mkdir()
    fe.PCC_dir = tmp_path / "pcc"
    fe.PCC_dir.mkdir()
    fe.script_dir = tmp_path / "scripts"
    fe.script_dir.mkdir()
    fe._check_done = lambda stage: False
    fe._set_done = lambda stage: None
    fe.pcc = SimpleNamespace(PCC_code="CODE")
    fe.target = SimpleNamespace()

    class DummyGMX:
        def __init__(self, *args, **kwargs):
            pass
        def create_topol(self):
            pass

    monkeypatch.setattr(fe_mod, "GMXitp", DummyGMX)
    monkeypatch.setattr(fe_mod, "subprocess", SimpleNamespace(run=_fake_run))

    with pytest.raises(RuntimeError):
        fe._mix()


def test_mix_uses_pcc_trjconv_output(tmp_path, monkeypatch):
    fe = FECalc.__new__(FECalc)
    fe.complex_dir = tmp_path / "complex"
    fe.complex_dir.mkdir()
    fe.target_dir = tmp_path / "target"
    fe.target_dir.mkdir()
    fe.PCC_dir = tmp_path / "pcc"
    (fe.PCC_dir / "em").mkdir(parents=True)
    (fe.PCC_dir / "PCC.acpype").mkdir(parents=True)
    fe.script_dir = tmp_path / "scripts"
    fe.script_dir.mkdir()
    fe._check_done = lambda stage: False
    fe._set_done = lambda stage: None
    fe.pcc = SimpleNamespace(PCC_code="CODE")
    fe.target = SimpleNamespace()

    # create dummy input files
    for fname in ["MOL.itp", "MOL.pdb", "posre_MOL.itp"]:
        (fe.target_dir / fname).write_text("")
    for fname in ["PCC_GMX.itp", "posre_PCC.itp"]:
        (fe.PCC_dir / "PCC.acpype" / fname).write_text("")
    (fe.PCC_dir / "em" / "CODE_em.pdb").write_text("")

    class DummyGMX:
        def __init__(self, *args, **kwargs):
            pass

        def create_topol(self):
            pass

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    def fake_place(*args, **kwargs):
        Path(args[2]).touch()

    monkeypatch.setattr(fe_mod, "GMXitp", DummyGMX)
    monkeypatch.setattr(fe_mod, "subprocess", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(fe_mod, "_place_in_box", fake_place)

    fe._mix()

    assert ["cp", f"{fe.PCC_dir}/em/CODE_em.pdb", "./PCC.pdb"] in calls


def test_eq_complex_propagates_missing_em_gro(tmp_path, monkeypatch):
    fe = FECalc.__new__(FECalc)
    fe.complex_dir = tmp_path / "complex"
    fe.complex_dir.mkdir()
    fe.script_dir = tmp_path / "scripts"
    fe.script_dir.mkdir(parents=True, exist_ok=True)
    fe.pcc = SimpleNamespace(PCC_code="CODE", charge=0)
    fe.MOL_list = []
    fe.PCC_charge = 0
    fe.box_size = 1
    fe._check_done = lambda stage: False
    fe._set_done = lambda stage: None
    fe.update_mdp = lambda *args, **kwargs: None
    fe._fix_posre = lambda: None

    monkeypatch.setattr(fe_mod, "subprocess", SimpleNamespace(run=_fake_run))

    with pytest.raises(FileNotFoundError):
        fe._eq_complex()


def test_pbmetad_raises_when_md_gro_missing(tmp_path, monkeypatch):
    fe = FECalc.__new__(FECalc)
    fe.complex_dir = tmp_path / "complex"
    fe.complex_dir.mkdir()
    fe.script_dir = tmp_path / "scripts"
    fe.script_dir.mkdir()
    fe.pcc = SimpleNamespace(PCC_code="CODE")
    fe.PCC_charge = 0
    fe.metad_bias_factor = 1
    fe.n_steps = 1000
    fe.metad_pace = 1
    fe.T = 300
    fe.metad_height = 1
    fe._create_plumed = lambda *args, **kwargs: None
    fe.update_mdp = lambda *args, **kwargs: None
    fe._set_done = lambda stage: None

    monkeypatch.setattr(fe_mod, "subprocess", SimpleNamespace(run=_fake_run))

    with pytest.raises(RuntimeError):
        fe._pbmetaD()


def test_reweight_propagates_subprocess_error(tmp_path, monkeypatch):
    fe = FECalc.__new__(FECalc)
    fe.complex_dir = tmp_path / "complex"
    md_dir = fe.complex_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)
    fe.script_dir = tmp_path / "scripts"
    fe.script_dir.mkdir()
    fe._create_plumed = lambda *args, **kwargs: None
    fe._set_done = lambda stage: None
    fe.pcc = SimpleNamespace(PCC_code="CODE")

    def fail_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(fe_mod, "subprocess", SimpleNamespace(run=fail_run))

    with pytest.raises(subprocess.CalledProcessError):
        fe._reweight()
