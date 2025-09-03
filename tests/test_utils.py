import sys
from pathlib import Path
import numpy as np
import pytest

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.utils import _read_pdb, _write_coords_to_pdb, _prep_pdb, cd, _place_in_box


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


def test_read_pdb_ignores_non_atom_records(tmp_path):
    pdb_content = (
        "REMARK comment line\n"
        "ATOM      1  C   MOL     1       0.000   0.000   0.000\n"
        "TER\n"
    )
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)

    mols, atoms, coords = _read_pdb(pdb_file)

    assert list(mols) == ["MOL"]
    assert list(atoms) == ["C"]
    assert np.allclose(coords, [[0.0, 0.0, 0.0]])


def test_read_pdb_empty_file(tmp_path):
    pdb_file = tmp_path / "empty.pdb"
    pdb_file.write_text("")

    mols, atoms, coords = _read_pdb(pdb_file)

    assert len(mols) == len(atoms) == len(coords) == 0


def test_read_pdb_invalid_path():
    with pytest.raises(FileNotFoundError):
        _read_pdb("nonexistent.pdb")


def test_read_pdb_malformed_line(tmp_path):
    # missing z coordinate field
    pdb_content = "ATOM      1  C   MOL     1       0.000   0.000\n"
    pdb_file = tmp_path / "bad.pdb"
    pdb_file.write_text(pdb_content)

    with pytest.raises((IndexError, ValueError)):
        _read_pdb(pdb_file)


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


def test_write_coords_to_pdb_mismatched_length(tmp_path):
    pdb_content = (
        "ATOM      1  C   UNL     1       0.000   0.000   0.000\n"
        "ATOM      2  H   UNL     1       0.100   0.100   0.100\n"
    )
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    coords = np.array([[1.0, 2.0, 3.0]])

    with pytest.raises(IndexError):
        _write_coords_to_pdb(infile, outfile, coords)


def test_write_coords_to_pdb_invalid_shape(tmp_path):
    pdb_content = "ATOM      1  C   UNL     1       0.000   0.000   0.000\n"
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    coords = np.array([1.0, 2.0, 3.0])

    with pytest.raises(IndexError):
        _write_coords_to_pdb(infile, outfile, coords)


def test_write_coords_to_pdb_formats_negative_and_precision(tmp_path):
    pdb_content = "ATOM      1  C   UNL     1       0.000   0.000   0.000\n"
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    original = infile.read_text().splitlines()[0]
    coords = np.array([[-1.23456, 123.4567, -0.0004]])

    _write_coords_to_pdb(infile, outfile, coords)

    line = outfile.read_text().splitlines()[0]
    assert line[:30] == original[:30]
    assert line[54:] == original[54:]
    assert line[30:38].strip() == "-1.235"
    assert line[38:46].strip() == "123.457"
    assert line[46:54].strip() == "-0.000"


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


def test_prep_pdb_preserves_other_records(tmp_path):
    pdb_content = (
        "ATOM      1  C   UNL     2       0.000   0.000   0.000\n"
        "TER\n"
        "REMARK something\n"
    )
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    _prep_pdb(infile, outfile, "NEW")

    lines = outfile.read_text().splitlines()
    assert "TER" in lines
    assert "REMARK something" in lines


def test_prep_pdb_strips_resname_whitespace(tmp_path):
    pdb_content = "ATOM      1  C   UNL     2       0.000   0.000   0.000\n"
    infile = tmp_path / "in.pdb"
    outfile = tmp_path / "out.pdb"
    infile.write_text(pdb_content)

    _prep_pdb(infile, outfile, " N E W ")

    line = outfile.read_text().splitlines()[0]
    assert line[17:20] == "NEW"


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


def test_cd_file_path_raises_value_error(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    start = Path.cwd()
    with pytest.raises(ValueError):
        with cd(file_path):
            pass
    assert Path.cwd() == start


def test_cd_nested_contexts(tmp_path):
    start = Path.cwd()
    dir1 = tmp_path / "dir1"
    dir2 = dir1 / "dir2"
    dir2.mkdir(parents=True)

    with cd(dir1):
        assert Path.cwd() == dir1
        with cd(dir2):
            assert Path.cwd() == dir2
        assert Path.cwd() == dir1
    assert Path.cwd() == start


def test_place_in_box_separates_and_aligns(tmp_path):
    pcc_content = (
        "ATOM      1  C   PCC     1       0.000   0.000   0.000\n"
        "ATOM      2  C   PCC     1       1.000   0.000   0.000\n"
        "ATOM      3  C   PCC     1       0.000   1.000   0.000\n"
        "TER\n"
    )
    mol_content = (
        "ATOM      1  C   MOL     1       0.000   0.000   0.000\n"
        "ATOM      2  C   MOL     1       0.000   1.000   0.000\n"
        "ATOM      3  C   MOL     1       0.000   0.000   1.000\n"
        "TER\n"
    )
    pcc_file = tmp_path / "PCC.pdb"
    mol_file = tmp_path / "MOL.pdb"
    out_file = tmp_path / "complex.pdb"
    pcc_file.write_text(pcc_content)
    mol_file.write_text(mol_content)

    _place_in_box(pcc_file, mol_file, out_file, box_size=20.0)

    res, atoms, coords = _read_pdb(out_file)
    pcc_coords = coords[res == "PCC"]
    mol_coords = coords[res == "MOL"]

    def normal(c):
        c0 = c - c.mean(axis=0)
        _, _, vh = np.linalg.svd(c0)
        return vh[2] / np.linalg.norm(vh[2])

    n1 = normal(pcc_coords)
    n2 = normal(mol_coords)
    assert abs(np.dot(n1, n2)) > 0.999

    diff = pcc_coords[:, None, :] - mol_coords[None, :, :]
    min_d = np.min(np.linalg.norm(diff, axis=2))
    assert min_d > 3.0 - 1e-6

    center_sep = np.linalg.norm(pcc_coords.mean(axis=0) - mol_coords.mean(axis=0))
    assert center_sep >= 10.0

    assert coords.min() >= 0.0
    assert coords.max() <= 20.0
