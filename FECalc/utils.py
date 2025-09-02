import os
from pathlib import Path

import numpy as np

def _read_pdb(fname):
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
            atom_coordinates.append(np.array([float(line_list[5]), float(line_list[6]), float(line_list[7])]))
    return np.array(molecule_types), np.array(atom_types), np.array(atom_coordinates)

def _write_coords_to_pdb(f_in, f_out, coords):
    with open(f_in) as f:
        f_cnt = f.readlines()
    
    new_file = []
    atom_i = 0
    for line in f_cnt:
        line_list = line.split()
        if line_list[0] == "ATOM" or line_list[0] == "HETATM":
            x = f"{coords[atom_i, 0]:.3f}"
            x = " "*(8-len(x)) + x
            y = f"{coords[atom_i, 1]:.3f}"
            y = " "*(8-len(y)) + y
            z = f"{coords[atom_i, 2]:.3f}"
            z = " "*(8-len(z)) + z
            new_line = line[:30]+x+y+z+line[54:]
            atom_i += 1
        else:
            new_line = line
    
        new_file.append(new_line)
    
    with open(f_out, "w") as f:
        f.writelines(new_file)

    return None

def _prep_pdb(in_f_dir: Path, out_f_dir: Path, resname: str) -> None:
    """
    acpype only accepts pdb files with one residue. Remove all residue information from acpype input file.

    Returns:
        None
    """
    # clean resname
    resname = "".join(resname.split())
    with open(f"{in_f_dir}") as f:
        pdb_cnt = f.readlines()
    line_identifier = ['HETATM', 'ATOM']
    acpype_pdb = []
    for line in pdb_cnt:
        line_list = line.split()
        if line_list[0] in line_identifier:
            new_line = line[:17]+f"{resname.strip()}"+line[20:25]+"1"+line[26:]
        else:
            new_line = line
        acpype_pdb.append(new_line)
        
    with open(f"{out_f_dir}", 'w') as f:
        f.writelines(acpype_pdb)
        
    return None

class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = Path(newPath)

    def __enter__(self):
        self.savedPath = Path.cwd()
        try:
            os.chdir(self.newPath)
        except (FileNotFoundError, NotADirectoryError) as e:
            raise ValueError("Path does not exist.") from e

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
