import sys
from pathlib import Path
import numpy as np

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.utils import _read_pdb, _write_coords_to_pdb, _prep_pdb, cd


def test_read_pdb_parses_atoms(tmp_path):
    pdb_content = (
        "HETATM    1  H   UNL     1      -0.038   1.930   0.661\n"
        "ATOM      2  C   MOL     1       0.000   0.000   0.000\n"
    )
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)

    mols, atoms, coords = _read_pdb(pdb_file)

    assert list(mols) == ["UNL", "MOL"]
    assert list(atoms) == ["H", "C"]
    assert np.allclose(coords, [[-0.038, 1.930, 0.661], [0.0, 0.0, 0.0]])


def test_write_coords_to_pdb_updates_coordinates(tmp_path):
    pdb_in_content = (
        "HETATM    1  H   UNL     1      -0.038   1.930   0.661\n"
        "ATOM      2  C   UNL     1       0.000   0.000   0.000\n"
        "TER\n"
    )
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_in_content)

    new_coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _write_coords_to_pdb(infile, outfile, new_coords)

    lines = outfile.read_text().splitlines()
    assert lines[0][30:38].strip() == "1.000"
    assert lines[0][38:46].strip() == "2.000"
    assert lines[0][46:54].strip() == "3.000"
    assert lines[1][30:38].strip() == "4.000"
    assert lines[1][38:46].strip() == "5.000"
    assert lines[1][46:54].strip() == "6.000"
    assert lines[2] == "TER"


def test_prep_pdb_rewrites_residue(tmp_path):
    pdb_content = (
        "HETATM    1  H   UNL     2      -0.038   1.930   0.661\n"
        "ATOM      2  C   UNL     3       0.000   0.000   0.000\n"
    )
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    _prep_pdb(infile, outfile, "NEW")

    for line in outfile.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            assert line[17:20] == "NEW"
            assert line[25] == "1"


def test_cd_changes_and_restores_directory(tmp_path):
    start = Path.cwd()
    with cd(tmp_path):
        assert Path.cwd() == tmp_path
    assert Path.cwd() == start

    non_existing = tmp_path / "missing"
    try:
        with cd(non_existing):
            pass
    except ValueError:
        assert Path.cwd() == start
    else:
        assert False, "cd should raise ValueError for invalid path"
