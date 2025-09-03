import argparse
import re
from pathlib import Path


class GMXitp():
    """Merge ``itp`` files into a single topology for complex simulations."""

    def __init__(self, MOL_itp, PCC_itp) -> None:
        """Store paths to the molecule and peptide ``itp`` files."""
        self.MOL_itp_dir = Path(MOL_itp)
        self.PCC_itp_dir = Path(PCC_itp)
        self.base_dir = self.MOL_itp_dir.parent

        # defining all the present atom_types
        self.atom_types = []  # holds the atom types
        self.atomtypes_sec = []  # hold the complete parameter line

    def _load_itp(self, itp_dir) -> set:
        """Extract the ``[ atomtypes ]`` section from an ``itp`` file."""

        with open(itp_dir) as f:
            itp_cnt = f.readlines()

        atomtypes_list = []  # final dictionary of sections
        start_line = None
        end_line = None

        for i, line in enumerate(itp_cnt):  # get the starting lines first
            if re.search("\[.*[a-z]+.*\]", line):
                title = "".join(line.split())
                title = title[1:-1]
                if not (start_line is None):
                    end_line = i - 1
                    break
                if title == "atomtypes":
                    start_line = i + 2

        atomtypes_list = itp_cnt[start_line:end_line]
        itp_truncated = itp_cnt[: start_line - 2] + itp_cnt[end_line:]

        return atomtypes_list, itp_truncated

    def _update_atomtypes(self, itp_atomtypes) -> None:
        """Collect unique atom types from an ``itp`` file."""
        for atom in itp_atomtypes:  # add them to the objects holding the list
            atom_list = atom.split()
            if len(atom_list) > 2:  # prevent empty spaces from breaking the code
                atom_type = atom_list[0]
            else:
                continue
            if atom_type not in self.atom_types:
                self.atom_types.append(atom_type.lower())
                self.atomtypes_sec.append(atom)
        return None

    def _make_top(self) -> None:
        """Write a combined ``topol.top`` for the complex."""
        content = [
            "; topol.top\n",
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
        with open(self.base_dir / "topol.top", "w") as f:
            f.writelines(content)

        return None

    def _make_complex_itp(self) -> None:
        """Create ``complex.itp`` containing all unique atom types."""
        content = [
            "; complex.itp\n",
            "\n",
            "[ atomtypes ]\n",
            ";name   bond_type     mass     charge   ptype   sigma         epsilon       Amb\n",
        ]

        content += self.atomtypes_sec

        content.append("\n")

        with open(self.base_dir / "complex.itp", "w") as f:
            f.writelines(content)

        return None

    def create_topol(self) -> None:
        """Generate truncated ``itp`` files and the combined topology."""
        PCC_atomtypes, PCC_trunc = self._load_itp(self.PCC_itp_dir)
        self._update_atomtypes(PCC_atomtypes)

        MOL_atomtypes, MOL_trunc = self._load_itp(self.MOL_itp_dir)
        self._update_atomtypes(MOL_atomtypes)

        # make complex itp from self.atomtypes_sec
        self._make_complex_itp()

        # write the truncated MOL and PCC itp files
        with open(self.base_dir / "MOL_truncated.itp", "w") as f:
            f.writelines(MOL_trunc)

        with open(self.base_dir / "PCC_truncated.itp", "w") as f:
            f.writelines(PCC_trunc)

        # make topol.top and include the forcefield.itp
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
