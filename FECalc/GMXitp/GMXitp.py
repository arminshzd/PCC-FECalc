import argparse
import re
from pathlib import Path

class GMXitp():
    """Combine MOL and PCC ``.itp`` files into a single topology."""

    def __init__(self, MOL_itp, PCC_itp) -> None:
        """Initialize with paths to the component ``.itp`` files."""
        self.MOL_itp_dir = Path(MOL_itp)
        self.PCC_itp_dir = Path(PCC_itp)
        self.base_dir = self.MOL_itp_dir.parent

        self.atom_types = []  # holds the atom types
        self.atomtypes_sec = []  # hold the complete parameter line

    def _load_itp(self, itp_dir) -> set:
        """Extract the ``[ atomtypes ]`` section from an ``.itp`` file."""

        with open(itp_dir) as f:
            itp_cnt = f.readlines()
        
        atomtypes_list = [] # final dictionary of sections
        start_line = None
        end_line = None

        for i, line in enumerate(itp_cnt): # get the starting lines first
            if re.search("\[.*[a-z]+.*\]", line):
                title = "".join(line.split())
                title = title[1:-1]
                if not(start_line is None):
                    end_line = i-1
                    break
                if title=="atomtypes":
                    start_line = i+2

        atomtypes_list = itp_cnt[start_line: end_line]
        itp_truncated = itp_cnt[:start_line-2] + itp_cnt[end_line:]
        
        return atomtypes_list, itp_truncated

    def _update_atomtypes(self, itp_atomtypes) -> None:
        """Update internal atom type lists with entries from an ``.itp`` file."""
        for atom in itp_atomtypes:
            atom_list = atom.split()
            if len(atom_list) > 2:
                atom_type = atom_list[0]
            else:
                continue
            if atom_type not in self.atom_types:
                self.atom_types.append(atom_type.lower())
                self.atomtypes_sec.append(atom)
        return None
    
    def _make_top(self) -> None:
        """Write ``topol.top`` including all required ``.itp`` files."""
        content = ["; topol.top\n",
                   "\n",
                   "; Include AMBER\n",
                   """#include "amber99sb.ff/forcefield.itp"\n""",
                   "\n",
                   "; Include complex.itp topology\n",
                   """#include "complex.itp"\n""",
                   "\n",
                   "; Include PCC.itp topology\n",
                   """#include "PCC_truncated.itp"\n""",
                   "\n",
                   "; PCC position restraints\n",
                   "#ifdef POSRES_PCC\n",
                   """#include "posre_PCC.itp"\n""",
                   "#endif\n",
                   "\n",
                   "; Include MOL.itp topology\n",
                   """#include "MOL_truncated.itp"\n""",
                   "\n",
                   "; MOL position restraints\n",
                   "#ifdef POSRES_MOL\n",
                   """#include "posre_MOL.itp"\n""",
                   "#endif\n",
                   "\n",
                   "; Include ions.itp\n",
                   """#include "amber99sb.ff/ions.itp"\n""",
                   "\n",
                   "; Include water topology\n",
                   """#include "amber99sb.ff/tip3p.itp"\n""",
                   "\n",
                   "[ system ]\n",
                   "Complex in water\n",
                   "\n",
                   "[ molecules ]\n",
                   "; Compound        nmols\n",
                   "PCC            1\n",
                   "MOL              1\n",
                   "\n",
                   ]
        with open(self.base_dir/"topol.top", "w") as f:
            f.writelines(content)
        
        return None

    def _make_complex_itp(self) -> None:
        """Write ``complex.itp`` containing the unified atom types."""
        content = ["; complex.itp\n",
                   "\n",
                   "[ atomtypes ]\n", 
                   ";name   bond_type     mass     charge   ptype   sigma         epsilon       Amb\n"]
    
        content += self.atomtypes_sec

        content.append("\n")

        with open(self.base_dir/"complex.itp", 'w') as f:
            f.writelines(content)
        
        return None

    def create_topol(self) -> None:
        """Generate combined topology and truncated ``.itp`` files."""
        PCC_atomtypes, PCC_trunc = self._load_itp(self.PCC_itp_dir)
        self._update_atomtypes(PCC_atomtypes)

        MOL_atomtypes, MOL_trunc = self._load_itp(self.MOL_itp_dir)
        self._update_atomtypes(MOL_atomtypes)

        self._make_complex_itp()

        with open(self.base_dir/"MOL_truncated.itp", 'w') as f:
            f.writelines(MOL_trunc)

        with open(self.base_dir/"PCC_truncated.itp", 'w') as f:
            f.writelines(PCC_trunc)

        self._make_top()

        return None

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-mi", "--molinput", help="MOL.itp file")
    parser.add_argument("-pi", "--pccinput", help="PCC.itp file")

    args = parser.parse_args()

    itp_gen = GMXitp(args.molinput,
                  args.pccinput)
    itp_gen.create_topol()
