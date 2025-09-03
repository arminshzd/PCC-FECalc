import os
from pathlib import Path

import numpy as np


def _read_pdb(fname):
    """Read atomic information from a PDB file.

    The parser extracts residue identifiers, atom names and Cartesian
    coordinates from standard ``ATOM`` or ``HETATM`` records.

    Args:
        fname (Path or str): Path to the PDB file.

    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): Arrays containing the
            molecule types, atom names and coordinates, respectively.
    """
    with open(fname) as f:
        f_cnt = f.readlines()

    molecule_types = []
    atom_types = []
    atom_coordinates = []
    for line in f_cnt:
        line_list = line.split()
        if line_list[0] == "ATOM" or line_list[0] == "HETATM":
            molecule_types.append(line_list[3])
            atom_types.append(line_list[2])
            atom_coordinates.append(
                np.array([float(line_list[5]), float(line_list[6]), float(line_list[7])])
            )
    return np.array(molecule_types), np.array(atom_types), np.array(atom_coordinates)


def _write_coords_to_pdb(f_in, f_out, coords):
    """Write new coordinates into an existing PDB template.

    Args:
        f_in (Path or str): Input PDB file providing the topology template.
        f_out (Path or str): Output PDB file with updated coordinates.
        coords (np.ndarray): ``(N, 3)`` array of Cartesian coordinates.

    Returns:
        None
    """
    with open(f_in) as f:
        f_cnt = f.readlines()

    new_file = []
    atom_i = 0
    for line in f_cnt:
        line_list = line.split()
        if line_list[0] == "ATOM" or line_list[0] == "HETATM":
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
    """Strip residue information so a PDB contains a single residue.

    ACPYPE requires input structures to be a single residue. This function
    rewrites residue identifiers in a PDB file so that every atom belongs to
    the same residue.

    Args:
        in_f_dir (Path): Path to the original PDB file.
        out_f_dir (Path): Destination path for the modified PDB file.
        resname (str): Three-letter residue name to assign.

    Returns:
        None
    """
    resname = "".join(resname.split())
    with open(f"{in_f_dir}") as f:
        pdb_cnt = f.readlines()
    line_identifier = ["HETATM", "ATOM"]
    acpype_pdb = []
    for line in pdb_cnt:
        line_list = line.split()
        if line_list[0] in line_identifier:
            new_line = line[:17] + f"{resname.strip()}" + line[20:25] + "1" + line[26:]
        else:
            new_line = line
        acpype_pdb.append(new_line)

    with open(f"{out_f_dir}", "w") as f:
        f.writelines(acpype_pdb)

    return None


class cd:
    """Context manager for temporarily changing the working directory."""

    def __init__(self, newPath):
        """Store the directory to switch into.

        Args:
            newPath (Path or str): Target working directory.
        """
        self.newPath = Path(newPath)

    def __enter__(self):
        """Change into the requested directory."""
        self.savedPath = Path.cwd()
        try:
            os.chdir(self.newPath)
        except (FileNotFoundError, NotADirectoryError) as e:
            raise ValueError("Path does not exist.") from e

    def __exit__(self, etype, value, traceback):
        """Return to the original working directory."""
        os.chdir(self.savedPath)
