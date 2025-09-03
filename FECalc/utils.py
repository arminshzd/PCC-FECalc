import os
from pathlib import Path
import subprocess
import inspect
from subprocess import CalledProcessError

import numpy as np

def _read_pdb(fname):
    """Read residue names, atom names, and coordinates from a PDB file.

    Args:
        fname (Path or str): Path to the PDB file.

    Returns:
        tuple: Arrays of residue names, atom names, and coordinates.
    """
    with open(fname) as f:
        f_cnt = f.readlines()

    molecule_types = []
    atom_types = []
    atom_coordinates = []
    for line in f_cnt:
        line_list = line.split()
        if line_list[0] in {"ATOM", "HETATM"}:
            molecule_types.append(line_list[3])
            atom_types.append(line_list[2])
            atom_coordinates.append(
                np.array([float(line_list[5]), float(line_list[6]), float(line_list[7])])
            )
    return np.array(molecule_types), np.array(atom_types), np.array(atom_coordinates)

def _write_coords_to_pdb(f_in, f_out, coords):
    """Write new coordinates into a PDB template file.

    Args:
        f_in (Path or str): Path to the template PDB file.
        f_out (Path or str): Output PDB file path.
        coords (np.ndarray): Array of shape (N, 3) with new coordinates.

    Returns:
        None
    """
    with open(f_in) as f:
        f_cnt = f.readlines()

    new_file = []
    atom_i = 0
    for line in f_cnt:
        line_list = line.split()
        if line_list[0] in {"ATOM", "HETATM"}:
            x = f"{coords[atom_i, 0]:.3f}"
            x = " " * (8 - len(x)) + x
            y = f"{coords[atom_i, 1]:.3f}"
            y = " " * (8 - len(y)) + y
            z = f"{coords[atom_i, 2]:.3f}"
            z = " " * (8 - len(z)) + z
            new_line = line[:30] + x + y + z + line[54:]
            atom_i += 1
        else:
            new_line = line

        new_file.append(new_line)

    with open(f_out, "w") as f:
        f.writelines(new_file)

    return None

def _prep_pdb(in_f_dir: Path, out_f_dir: Path, resname: str) -> None:
    """Create a single-residue PDB suitable for ``acpype``.

    ``acpype`` requires the input PDB to contain only one residue. This
    function strips all residue information from ``in_f_dir`` and writes the
    sanitized structure to ``out_f_dir`` with the residue name ``resname``.

    Args:
        in_f_dir (Path): Input PDB file.
        out_f_dir (Path): Output PDB file.
        resname (str): Residue name to apply.

    Returns:
        None
    """
    resname = "".join(resname.split())
    with open(f"{in_f_dir}") as f:
        pdb_cnt = f.readlines()
    line_identifier = ['HETATM', 'ATOM']
    acpype_pdb = []
    for line in pdb_cnt:
        line_list = line.split()
        if line_list[0] in line_identifier:
            new_line = line[:17] + f"{resname.strip()}" + line[20:25] + "1" + line[26:]
        else:
            new_line = line
        acpype_pdb.append(new_line)

    with open(f"{out_f_dir}", 'w') as f:
        f.writelines(acpype_pdb)

    return None

class cd:
    """Context manager for changing the current working directory."""

    def __init__(self, newPath):
        """Store the new working directory."""
        self.newPath = Path(newPath)

    def __enter__(self):
        """Change to the new working directory."""
        self.savedPath = Path.cwd()
        try:
            os.chdir(self.newPath)
        except (FileNotFoundError, NotADirectoryError) as e:
            raise ValueError("Path does not exist.") from e

    def __exit__(self, etype, value, traceback):
        """Return to the original working directory."""
        os.chdir(self.savedPath)


def run_gmx(cmd, **kwargs):
    """Run a ``gmx`` command with improved error reporting.

    Parameters
    ----------
    cmd : list or str
        Command passed to :func:`subprocess.run`.  If a string is provided it
        is executed with ``shell=True``.
    **kwargs : dict
        Additional keyword arguments forwarded to ``subprocess.run``.

    Raises
    ------
    RuntimeError
        If the command exits with a non-zero status.  The raised error includes
        the command, stdout and stderr to aid debugging.
    """

    kwargs.setdefault("check", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("capture_output", True)

    if isinstance(cmd, str):
        kwargs.setdefault("shell", True)

    subprocess_module = kwargs.pop("subprocess_module", None)
    if subprocess_module is None:
        frame = inspect.currentframe().f_back
        subprocess_module = frame.f_globals.get("subprocess", subprocess)

    try:
        return subprocess_module.run(cmd, **kwargs)
    except CalledProcessError as e:
        cmd_str = e.cmd if isinstance(e.cmd, str) else " ".join(e.cmd)
        stdout = e.stdout.strip() if e.stdout else ""
        stderr = e.stderr.strip() if e.stderr else ""
        msg = f"Command '{cmd_str}' failed with code {e.returncode}."
        if stdout:
            msg += f"\nSTDOUT:\n{stdout}"
        if stderr:
            msg += f"\nSTDERR:\n{stderr}"
        raise RuntimeError(msg) from e


