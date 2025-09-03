import sys
from pathlib import Path
import subprocess

# ensure package root on path for import
sys.path.append(str(Path(__file__).resolve().parents[1]))

from FECalc.GMXitp.GMXitp import GMXitp


def _write_itp(path, atomtype_line, extra_section="[ atoms ]\n1 X 1\n"):
    content = (
        "; header\n"
        "[ atomtypes ]\n"
        "; comment\n"
        f"{atomtype_line}\n"
        "\n"
        f"{extra_section}"
    )
    path.write_text(content)


def test_load_itp_extracts_atomtypes_and_truncates(tmp_path):
    itp = tmp_path / "test.itp"
    _write_itp(itp, "CA  c  12  0  A  1.9  0.1")

    gmx = GMXitp(itp, itp)
    atomtypes, truncated = gmx._load_itp(itp)

    assert atomtypes == ["CA  c  12  0  A  1.9  0.1\n"]
    truncated_str = "".join(truncated)
    assert "[ atomtypes ]" not in truncated_str
    assert "[ atoms ]" in truncated_str


def test_update_atomtypes_deduplicates_case_insensitive(tmp_path):
    gmx = GMXitp(tmp_path / "M.itp", tmp_path / "P.itp")
    entries = [
        "CA  c  12  0  A  1.9  0.1\n",
        "ca  c  12  0  A  1.9  0.1\n",
        "HB  h   1  0  A  1.0  0.2\n",
        "invalid\n",
    ]
    gmx._update_atomtypes(entries)

    assert gmx.atom_types == ["ca", "hb"]
    assert gmx.atomtypes_sec == [
        "CA  c  12  0  A  1.9  0.1\n",
        "HB  h   1  0  A  1.0  0.2\n",
    ]


def test_create_topol_writes_combined_files(tmp_path):
    mol_itp = tmp_path / "MOL.itp"
    pcc_itp = tmp_path / "PCC.itp"
    _write_itp(mol_itp, "CA  c  12  0  A  1.9  0.1")
    _write_itp(pcc_itp, "HB  h   1  0  A  1.0  0.2")

    gmx = GMXitp(mol_itp, pcc_itp)
    gmx.create_topol()

    complex_content = (tmp_path / "complex.itp").read_text()
    assert "CA  c  12  0  A  1.9  0.1" in complex_content
    assert "HB  h   1  0  A  1.0  0.2" in complex_content

    mol_trunc = (tmp_path / "MOL_truncated.itp").read_text()
    pcc_trunc = (tmp_path / "PCC_truncated.itp").read_text()
    assert "[ atomtypes ]" not in mol_trunc
    assert "[ atomtypes ]" not in pcc_trunc

    topol_lines = (tmp_path / "topol.top").read_text().splitlines()
    assert topol_lines.count("#include \"complex.itp\"") == 1
    pcc_idx = topol_lines.index("#include \"PCC_truncated.itp\"")
    mol_idx = topol_lines.index("#include \"MOL_truncated.itp\"")
    assert pcc_idx < mol_idx


def test_cli_invocation_generates_topology(tmp_path):
    mol_itp = tmp_path / "MOL.itp"
    pcc_itp = tmp_path / "PCC.itp"
    _write_itp(mol_itp, "CA  c  12  0  A  1.9  0.1")
    _write_itp(pcc_itp, "HB  h   1  0  A  1.0  0.2")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "FECalc.GMXitp.GMXitp",
            "-mi",
            str(mol_itp),
            "-pi",
            str(pcc_itp),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert (tmp_path / "topol.top").exists()
    assert (tmp_path / "complex.itp").exists()
