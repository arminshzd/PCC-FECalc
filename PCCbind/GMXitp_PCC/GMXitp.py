import argparse
import re
from pathlib import Path

class GMXitp():
    def __init__(self, PCC1_itp, PCC2_itp) -> None:
        # defining directories
        self.PCC1_itp_dir = Path(PCC1_itp)
        self.PCC2_itp_dir = Path(PCC2_itp)
        self.base_dir = self.PCC1_itp_dir.parent

        #defining all the present atom_types
        self.atom_types = [] # holds the atom types
        self.atomtypes_sec = [] # hold the complete parameter line
    
    def _rename_res(self, itp_trunc: list,oldres: str, newres: str) -> None:
        """ Rename the residue in the truncated itp.

        Args:
            itp_trunc (list): truncated list of strings
            oldres (str): resname to be replaced
            newres (str): new resname
        """
        for i, line in enumerate(itp_trunc):
            if oldres in line:
                itp_trunc[i] = line.replace(oldres, newres)
        return None
    
    def _load_itp(self, itp_dir, resname: str = None, newres: str = "UNK") -> set:
        """Doesn't really proces the lines. Just the sections. Optionally, replaces
        the residue name in the truncated file."""

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
        if resname is not None:
            self._rename_res(itp_truncated, resname, newres)
             
        return atomtypes_list, itp_truncated

    def _update_atomtypes(self, itp_atomtypes) -> None:
        for atom in itp_atomtypes: # add them to the objects holding the list of atom types
            atom_list = atom.split()
            if len(atom_list)>2: # prevent empty spaces from breaking the code
                atom_type = atom_list[0]
            else:
                continue
            if atom_type not in self.atom_types:
                self.atom_types.append(atom_type.lower())
                self.atomtypes_sec.append(atom)
        return None
    
    def _make_top(self) -> None:
        content = ["; topol.top\n",
                   "\n",
                   "; Include AMBER\n",
                   """#include "amber99sb.ff/forcefield.itp"\n""",
                   "\n",
                   "; Include complex.itp topology\n",
                   """#include "complex.itp"\n""",
                   "\n",
                   "; Include PCC1.itp topology\n",
                   """#include "PCC1_truncated.itp"\n""",
                   "\n",
                   "; PCC1 position restraints\n",
                   "#ifdef POSRES_PC1\n",
                   """#include "posre_PCC1.itp"\n""",
                   "#endif\n",
                   "\n",
		   "; Include PCC2.itp topology\n",
                   """#include "PCC2_truncated.itp"\n""",
                   "\n",
                   "; PCC2 position restraints\n",
                   "#ifdef POSRES_PC2\n",
                   """#include "posre_PCC2.itp"\n""",
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
                   "PC1            1\n",
                   "PC2            1\n",
                   "\n",
                   ]
        with open(self.base_dir/"topol.top", "w") as f:
            f.writelines(content)
        
        return None

    def _make_complex_itp(self) -> None:
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
        PCC1_atomtypes, PCC1_trunc = self._load_itp(self.PCC1_itp_dir, "PCC", "PC1")
        self._update_atomtypes(PCC1_atomtypes) # Update the two lists

        PCC2_atomtypes, PCC2_trunc = self._load_itp(self.PCC2_itp_dir, "PCC", "PC2")
        self._update_atomtypes(PCC2_atomtypes) # Update the two lists

        # make comlex itp from self.atomtypes_sec
        self._make_complex_itp()

        # write the truncated PCC1 and PCC2 itp files
        with open(self.base_dir/"PCC1_truncated.itp", 'w') as f:
            f.writelines(PCC1_trunc)
        
        with open(self.base_dir/"PCC2_truncated.itp", 'w') as f:
            f.writelines(PCC2_trunc)
        

        # make topol.top and include the forcefield.itp, 
        self._make_top()

        return None

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-pi",  "--pcc1input", help="PCC1.itp file")
    parser.add_argument("-ppi", "--pcc2input", help="PCC2.itp file")

    args = parser.parse_args()

    itp_gen = GMXitp(args.pcc1input,
                  args.pcc2input)
    itp_gen.create_topol()
