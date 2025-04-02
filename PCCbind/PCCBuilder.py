import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

import numpy as np


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = Path(newPath)

    def __enter__(self):
        self.savedPath = Path.cwd()
        try:
            os.chdir(self.newPath)
        except:
            raise ValueError("Path does not exist.")

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class PCCBuilder():
    """
    Class to generate the PCC from the master PCC structure. Then `AMBER` parameters
    are generated for the new PCC using `acpype`. New PCC is then solvated and equilibrated.
    """

    def __init__(self, pcc: str, base_dir: Path, settings_json: Path) -> None:
        """
        Setup the PCCBuilder directories
    
        Args:
            pcc (str): single letter string of AAs, ordered from arm to bead on the PCC structure.
            base_dir (Path): directory to store the calculations
            settings_json (Path): directory of settings.JSON file

        Raises:
            ValueError: Raises Value error if `base_dir` is not a directory.
        """
        self.PCC_code = pcc
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Building and minimizing structure for {self.PCC_code} (PID: {os.getpid()})")
        self.settings_dir = Path(settings_json) # path settings.JSON
        with open(self.settings_dir) as f:
            self.settings = json.load(f)
        self.script_dir = Path(__file__).parent.parent/"FECalc"/"scripts"
        self.mold_dir = Path(__file__).parent.parent/"FECalc"/"mold"
        base_dir = Path(base_dir)
        if base_dir.exists():
            if not base_dir.is_dir():
                raise ValueError(f"{base_dir} is not a directory.")
        else:
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Base directory does not exist. Creating...")
            base_dir.mkdir()        
        self.base_dir = base_dir # base directory to store files

        self.PCC_dir = self.base_dir # directory to store PCC calculations
        self.PCC_dir.mkdir(exist_ok=True)

        self.PCC_ref = self.script_dir/"FGGGG.pdb" # path to refrence PCC structure
        
        self.AAdict31 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'} # 3 to 1 translator
        self.AAdict13 = {j: i for i, j in self.AAdict31.items()} # 1 to 3 translator
        self.charge_dict = {"D": -1, "E": -1, "R": +1, "K": +1} # AA charges at neutral pH

        self.PCC_charge = sum([self.charge_dict.get(i, 0) for i in list(self.PCC_code)]) # Net charge of the PCC at pH=7
        self.PCC_n_atoms = None # number of PCC atoms

        self.pymol = Path(self.settings['pymol_dir']) # path to pymol installation
        self.acpype = Path(self.settings['acpype_dir']) # Path to acpype conda env

    def _check_done(self, stage: str) -> bool:
        """
        Check if calculation has been performed already
        Args:
            stage (str): calculation stage to check

        Returns:
            bool: True if done, False otherwise
        """
        try:
            with cd(self.base_dir/stage):
                done_dir = self.base_dir/stage/".done"
                if done_dir.exists():
                    return True
                else:
                    return False
        except:
            return False
        
    def _set_done(self, stage: str) -> None:
        """
        create an empty file ".done" in the stage directory to mark is has been performed already.

        Args:
            stage (str): stage to set as done

        Returns:
            None
        """
        with cd(self.base_dir/stage):
            done_dir = self.base_dir/stage/".done"
            with open(done_dir, 'w') as f:
                f.writelines([""])
        return None
    
    def _read_pdb(self, fname):
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

    def _write_coords_to_pdb(self, f_in, f_out, coords):
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

    def _create_pcc(self) -> None:
        """
        Call `PCCmold.py` through `pymol` to mutate all residues in the refrence PCC to create the new PCC.

        Returns:
            None
        """
        pcc_translate = [self.AAdict13[i] for i in list(self.PCC_code)] # translate the 1 letter AA code to 3 letter code
        numres = len(pcc_translate) # number of residues
        mutation = "".join(pcc_translate) # concatenate the 3 letter codes
        subprocess.run(f"{self.pymol} -qc {self.script_dir}/PCCmold.py -i {self.PCC_ref} -o {self.PCC_dir/self.PCC_code}.pdb -r {numres} -m {mutation}", shell=True, check=True)
	# pre-optimization
        wait_str = " --wait "
        subprocess.run(f"cp {self.mold_dir}/PCC/sub_preopt.sh {self.PCC_dir}", shell=True) # copy preopt submission script
        with cd(self.PCC_dir): # cd into the PCC directory
            # pre-optimize to deal with possible clashes created while changing residues to D-AAs
            print("Pre-optimizing: ", flush=True)
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_preopt.sh {self.PCC_code}.pdb {self.PCC_code}_babel.pdb", shell=True, check=True)
            _, _, coords_new = self._read_pdb(f"{self.PCC_code}_babel.pdb")
            self._write_coords_to_pdb(f"{self.PCC_code}.pdb", f"{self.PCC_code}_opt.pdb", coords_new[:self.PCC_n_atoms, ...])

        self._set_done(self.PCC_dir) # mark stage as done
        return None
    
    def _prep_pdb(self) -> None:
        """
        acpype only accepts pdb files with one residue. Remove all residue information from acpype input file.

        Returns:
            None
        """
        with cd(self.PCC_dir): # cd into the PCC directory
            with open(f"{self.PCC_code}_opt.pdb") as f:
                pdb_cnt = f.readlines()
            line_identifier = ['HETATM', 'ATOM']
            acpype_pdb = []
            for line in pdb_cnt:
                line_list = line.split()
                if line_list[0] in line_identifier:
                    new_line = line[:17]+"PCC"+line[20:25]+"1"+line[26:]
                else:
                    new_line = line
                acpype_pdb.append(new_line)
            
            with open(f"{self.PCC_code}_acpype.pdb", 'w') as f:
                f.writelines(acpype_pdb)
            
            return None
    
    def _get_params(self, wait: bool = True) -> None: 
        """
        Run acpype on the mutated `PCC.pdb` file. Submits a job to the cluster.

        Args:
            wait (bool, optional): Whether to wait for acpype to finish. Defaults to True.

        Returns:
            None
        """
        
        subprocess.run(f"cp {self.mold_dir}/PCC/sub_acpype.sh {self.PCC_dir}", shell=True) # copy acpype submission script

        wait_str = " --wait " if wait else "" # whether to wait for acpype to finish before exiting        
        with cd(self.PCC_dir): # cd into the PCC directory
                        # create acpype pdb with 1 residue
            self._prep_pdb()
            # run acpype
            print("Running acpype: ", flush=True)
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_acpype.sh {self.PCC_code}_acpype {self.PCC_charge}", shell=True, check=True)
        
            # check acpype.log for warnings
            with open("PCC.acpype/acpype.log") as f:
                acpype_log = f.read()
                if "warning:" in acpype_log.lower():
                    raise RuntimeError("""Acpype generated files are likely to have incorrect bonds. 
                                       Check the generated PCC structure before continueing.""")

        self._set_done(self.PCC_dir/"PCC.acpype")
        return None
    
    def _get_n_atoms(self, gro_dir: Path) -> None:
        """
        Get the number of PCC atoms from gro file

        Args:
            gro_dir (Path): path to PCC.gro file.
        """
        with open(gro_dir) as f:
            gro_cnt = f.readlines()
        self.PCC_n_atoms = int(gro_cnt[1].split()[0])
    
    def _minimize_PCC(self, wait: bool = True) -> None: 
        """
        Run minimization for PCC. Copies acpype files into `em` directory, solvates, adds ions, and minimizes
        the structure. The last frame is saved as `PCC.gro`

        Args:
            wait (bool, optional): Whether to wait for `em` to finish. Defaults to True.

        Returns:
            None
        """
        Path.mkdir(self.PCC_dir/"em", exist_ok=True)
        with cd(self.PCC_dir/"em"): # cd into PCC/em
            # copy acpype files into em dir
            subprocess.run("cp ../PCC.acpype/PCC_GMX.gro .", shell=True, check=True)
            subprocess.run("cp ../PCC.acpype/PCC_GMX.itp .", shell=True, check=True)
            subprocess.run("cp ../PCC.acpype/posre_PCC.itp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/topol.top .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/ions.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/em.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/sub_mdrun_PCC_em.sh .", shell=True) # copy mdrun submission script
            # set self.PCC_n_atoms
            self._get_n_atoms("./PCC_GMX.gro")
            # submit em job
            wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_PCC_em.sh {self.PCC_charge}", check=True, shell=True)
        self._set_done(self.PCC_dir/"em")

        return None

    def create(self) -> tuple:
        """Wrapper for building PCC structures.

        Returns:
            None
        """
        # create PCC
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Running pymol: ", end="", flush=True)
        if not self._check_done(self.PCC_dir):
            self._create_pcc()
            print("Check the initial structure and rerun to continue.")
            return None
        print("\tDone.", flush=True)
        # get params
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Getting gaff parameters: ", flush=True)
        if not self._check_done(self.PCC_dir/"PCC.acpype"):
            self._get_params()
        print("Done.", flush=True)
        # minimize PCC
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Minimizing PCC: ", end="", flush=True)
        if not self._check_done(self.PCC_dir/'em'):
            self._minimize_PCC()
        print("\tDone.", flush=True)
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: All steps completed.")
        print("-"*30 + "Finished" + "-"*30)
        
        return None