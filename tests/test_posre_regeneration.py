import os
import sys
from pathlib import Path

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.FECalc import FECalc


def test_fix_posre_regenerates_indices(tmp_path):
    """_fix_posre should write the atom ids obtained from _get_atom_ids."""

    # create a minimal gro file with known atom ids
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

    # create FECalc instance without invoking __init__
    fe = FECalc.__new__(FECalc)
    fe.MOL_list = []
    fe.PCC_list = []
    fe.MOL_list_atom = []
    fe.PCC_list_atom = []

    # populate atom lists and regenerate posre files
    fe._get_atom_ids(gro_file)
    os.chdir(tmp_path)
    fe._fix_posre()

    # read generated position restraint files
    mol_lines = [
        line
        for line in (tmp_path / "posre_MOL.itp").read_text().splitlines()
        if line and not line.startswith(";")
    ]
    pcc_lines = [
        line
        for line in (tmp_path / "posre_PCC.itp").read_text().splitlines()
        if line and not line.startswith(";")
    ]

    mol_ids = [int(line.split()[0]) for line in mol_lines[1:]]
    pcc_ids = [int(line.split()[0]) for line in pcc_lines[1:]]

    assert mol_ids == fe.MOL_list
    assert pcc_ids == fe.PCC_list

