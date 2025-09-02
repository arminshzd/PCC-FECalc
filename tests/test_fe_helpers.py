import sys
from pathlib import Path
import pytest

# ensure package root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.FECalc import FECalc


def test_check_and_set_done_handles_paths(tmp_path):
    fe = FECalc.__new__(FECalc)
    fe.base_dir = tmp_path

    # relative path
    assert not fe._check_done(Path("rel_stage"))
    fe._set_done(Path("rel_stage"))
    assert (tmp_path / "rel_stage" / ".done").exists()
    assert fe._check_done(Path("rel_stage"))

    # absolute path
    abs_stage = tmp_path / "abs_stage"
    assert not fe._check_done(abs_stage)
    fe._set_done(abs_stage)
    assert (abs_stage / ".done").exists()
    assert fe._check_done(abs_stage)


def test_get_atom_ids_reads_gro(tmp_path):
    gro_content = (
        "test\n"
        "4\n"
        "1MOL C1 1 0 0 0\n"
        "1MOL C2 2 0 0 0\n"
        "1PCC N1 3 0 0 0\n"
        "1PCC H1 4 0 0 0\n"
        "0 0 0\n"
    )
    gro_file = tmp_path / "em.gro"
    gro_file.write_text(gro_content)

    fe = FECalc.__new__(FECalc)
    fe._get_atom_ids(gro_file)

    assert fe.MOL_list == [1, 2]
    assert fe.PCC_list == [3, 4]
    assert fe.MOL_list_atom == ["C1", "C2"]
    assert fe.PCC_list_atom == ["N1", "H1"]


def test_get_atom_ids_missing_file(tmp_path):
    fe = FECalc.__new__(FECalc)
    with pytest.raises(FileNotFoundError):
        fe._get_atom_ids(tmp_path / "missing.gro")
