import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.FECalc import FECalc
from FECalc.postprocess import postprocess


def test_minimal_end_to_end(tmp_path, monkeypatch):
    # stub PCC object with required attributes
    pcc_dir = tmp_path / "pcc"
    pcc_dir.mkdir()
    pcc = types.SimpleNamespace(
        PCC_code="AAA",
        PCC_dir=pcc_dir,
        charge=0,
        origin=["N1"],
        anchor_point1=["C1"],
        anchor_point2=["C2"],
    )

    # stub target object
    target_base = tmp_path / "mol"
    (target_base / "export").mkdir(parents=True)
    target = types.SimpleNamespace(
        name="MOL",
        base_dir=target_base,
        anchor_point1=["C"],
        anchor_point2=["N"],
    )

    # patch FECalc methods to avoid external dependencies
    def fake_mix(self):
        self.complex_dir.mkdir(parents=True, exist_ok=True)
        (self.complex_dir / ".done").touch()

    def fake_eq_complex(self, wait=True):
        em_dir = self.complex_dir / "em"
        em_dir.mkdir(parents=True, exist_ok=True)
        em_dir.joinpath("em.gro").write_text(
            "test\n1\n1MOL C1 1 0 0 0\n1 1 1\n"
        )
        (em_dir / ".done").touch()
        for stage in ["nvt", "npt"]:
            sdir = self.complex_dir / stage
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / ".done").touch()

    def fake_pbmetad(self, wait=True):
        md_dir = self.complex_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)
        md_dir.joinpath("md.gro").write_text(
            "test\n1\n1MOL C1 1 0 0 0\n1 1 1\n"
        )
        md_dir.joinpath("GRID_COM").write_text("0\n")
        md_dir.joinpath("GRID_ang").write_text("0\n")
        md_dir.joinpath("GRID_cos").write_text("0\n")
        (md_dir / ".done").touch()

    def fake_reweight(self, wait=True):
        re_dir = self.complex_dir / "reweight"
        re_dir.mkdir(parents=True, exist_ok=True)
        re_dir.joinpath("COLVAR").write_text(
            "#! FIELDS time dcom ang v3cos pb.bias\n"
            "0 0.5 0.1 0.2 0\n"
            "1000 2.1 0.3 -0.2 0\n"
        )
        (re_dir / ".done").touch()

    monkeypatch.setattr(FECalc, "_mix", fake_mix)
    monkeypatch.setattr(FECalc, "_eq_complex", fake_eq_complex)
    monkeypatch.setattr(FECalc, "_pbmetaD", fake_pbmetad)
    monkeypatch.setattr(FECalc, "_reweight", fake_reweight)

    out_dir = tmp_path / "calc"
    calc = FECalc(pcc, target, out_dir, temp=300, box=5.0)
    calc.run(n_steps=1)
    postprocess(calc, discard_initial=0, n_folds=1)

    assert (out_dir / "metadata.JSON").exists()
