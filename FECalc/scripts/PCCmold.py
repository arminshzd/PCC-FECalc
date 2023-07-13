import sys
import argparse

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file")
parser.add_argument("-o", "--output", help="Output file")
parser.add_argument("-r", "--residlen", help="length of mutation section")
parser.add_argument("-m", "--mutant", help="Residue to mutate to")
parser.add_argument("-qc", "--quite", help="Pymol quite mode")

args = parser.parse_args()

residlen = args.residlen
residnums = [str(i+2) + "/" for i in range(int(residlen))]

mutant = args.mutant
if len(mutant)%3 != 0:
    raise ValueError("Wrong mutant input.")

mutations = [mutant[i*3:(i+1)*3] for i in range(len(mutant)//3)]

if len(mutations) != len(residnums):
    raise ValueError("Mutation length is not consistent with input residue length.")

# Load Structures
cmd.load(args.input, "MAIN")

# Call wizard
cmd.wizard("mutagenesis")
cmd.do("refresh_wizard")

for i, resid in enumerate(residnums):
    # Mutate residue
    print(f"Mutating residue {resid} to {mutations[i]}")
    cmd.get_wizard().set_mode(mutations[i]) # mutate to this
    cmd.get_wizard().do_select(resid) # at this location. "/" is necessary

    # Check rotamer scores
    scores = cmd.get_wizard().bump_scores
    if len(scores) != 0:
        min_ind = scores.index(min(scores))

        # Select the rotamer
        cmd.frame(min_ind+1)

    # Apply the mutation
    cmd.get_wizard().apply()

# invert mutated residues to get D-AAs
for i, resid in enumerate(residnums):
    print(f"Inverting residue {resid}")
    cmd.edit(resid+"ca", resid+"n", resid+"c")
    cmd.invert()

# Save new mutant
cmd.save(args.output, "all")

# Exit wizard
cmd.wizard(None)

# Get out!
cmd.quit()
