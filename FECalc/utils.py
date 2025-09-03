import os
from pathlib import Path

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


def _place_in_box(pcc_pdb, mol_pdb, out_pdb, box_size, min_dist=3.0, initial_sep=10.0):
    """Place PCC and target molecules into a simulation box without clashes.

    The two molecules are rotated so that their best-fit planes are parallel
    and separated along the z-axis. The separation between their geometric
    centers starts at ``initial_sep`` and is increased until the minimum
    interatomic distance between the two molecules exceeds ``min_dist``.

    Args:
        pcc_pdb (Path or str): Path to the PCC PDB file.
        mol_pdb (Path or str): Path to the target molecule PDB file.
        out_pdb (Path or str): Output PDB file for the combined complex.
        box_size (float): Edge length of the (cubic) simulation box in Å.
        min_dist (float, optional): Minimum allowed distance between any PCC
            and target atom in Å. Defaults to 3.0.
        initial_sep (float, optional): Initial separation between the
            geometric centers in Å. Defaults to 10.0 (1 nm).

    Returns:
        None
    """

    def _plane_normal(coords):
        coords_centered = coords - coords.mean(axis=0)
        _, _, vh = np.linalg.svd(coords_centered)
        return vh[2]

    def _align_to_xy(coords):
        n = _plane_normal(coords)
        n = n / np.linalg.norm(n)
        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(n, z_axis)
        s = np.linalg.norm(v)
        c = np.dot(n, z_axis)
        if s < 1e-8:
            if c > 0:
                R = np.eye(3)
            else:
                R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            v /= s
            vx = np.array(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
            )
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return coords @ R.T

    # read coordinates
    pcc_res, pcc_atoms, pcc_coords = _read_pdb(pcc_pdb)
    mol_res, mol_atoms, mol_coords = _read_pdb(mol_pdb)

    # align planes and center
    pcc_coords = _align_to_xy(pcc_coords)
    mol_coords = _align_to_xy(mol_coords)
    pcc_coords -= pcc_coords.mean(axis=0)
    mol_coords -= mol_coords.mean(axis=0)

    # increase separation until no clashes
    axis = np.array([0.0, 0.0, 1.0])
    sep = initial_sep
    while True:
        mol_shifted = mol_coords + axis * sep
        diff = pcc_coords[:, None, :] - mol_shifted[None, :, :]
        min_d = np.min(np.linalg.norm(diff, axis=2))
        if min_d >= min_dist:
            break
        sep += 1.0

    box_center = np.array([box_size / 2] * 3)
    pcc_coords_box = pcc_coords + box_center - axis * sep / 2
    mol_coords_box = mol_shifted + box_center - axis * sep / 2

    # write combined PDB
    def _update_lines(fname, coords):
        with open(fname) as f:
            lines = f.readlines()
        new_lines = []
        atom_i = 0
        for line in lines:
            fields = line.split()
            if fields and fields[0] in {"ATOM", "HETATM"}:
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
            new_lines.append(new_line)
        return new_lines

    pcc_lines = _update_lines(pcc_pdb, pcc_coords_box)
    mol_lines = _update_lines(mol_pdb, mol_coords_box)

    with open(out_pdb, "w") as f:
        f.writelines(pcc_lines + mol_lines)

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

