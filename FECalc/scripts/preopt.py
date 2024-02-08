import argparse

from ase.io import read as ase_read
from ase.optimize import BFGS
from ase.calculators.emt import EMT

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file")
parser.add_argument("-o", "--output", help="Output file")

args = parser.parse_args()

def write_coords_to_pdb(f_in, f_out, coords):
    """
    Function to write an identical pdb file ONLY changing the coordinates

    Args:
        f_in (str): dir of input pdb
        f_out (str): dir of output pdb
        coords (np.Array): array of new positions

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

PCC_struct = ase_read(args.input)
PCC_struct.set_calculator(EMT())
PCC_dyn = BFGS(PCC_struct)
PCC_dyn.run(fmax=0.1, steps=500)
write_coords_to_pdb(args.input, args.output, PCC_struct.get_positions())