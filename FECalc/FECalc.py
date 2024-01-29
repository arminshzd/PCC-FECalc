import os
import re
import subprocess
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.integrate import simpson

from .GMXitp.GMXitp import GMXitp

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

class FECalc():
    """
    Class to calculate free energy surface given PCC peptide chain and target name.
    First, the PCC is generated from the master PCC structure. Then `AMBER` parameters
    are generated for the new PCC using `acpype`. New PCC is then solvated and equilibrated.
    The equilibrated structures is placed in a box with the target molecule using `packmol`.
    The complex box is then solvated and equilibrated and the free energy surface is calculated
    using `PBMetaD`.
    """
    def __init__(self, pcc: str, target: str, base_dir: Path, settings_json: Path) -> None:
        """
        Setup the base, PCC, and complex directories, and locate the target molecule files.
    
        Args:
            pcc (str): single letter string of AAs, ordered from arm to bead on the PCC structure.
            target (str): 'FEN' or 'DEC'. Target molecule for FE calculations.
            base_dir (Path): directory to store the calculations
            script_dir (Path): directory containing necessary scripts and templates

        Raises:
            ValueError: Raises Value error if `base_dir` is not a directory.
        """
        self.settings_dir = Path(settings_json) # path settings.JSON
        with open(self.settings_dir) as f:
            self.settings = json.load(f)
        self.script_dir = Path(__file__).parent/Path("scripts")#Path(self.settings['scripts_dir'])
        self.mold_dir = Path(__file__).parent/Path("mold")
        self.PCC_code = pcc
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

        self.PCC_dir = self.base_dir/f"{self.PCC_code}" # directory to store PCC calculations
        self.PCC_dir.mkdir(exist_ok=True)

        self.complex_dir = self.base_dir/"complex" # directory to store complex calculations
        self.complex_dir.mkdir(exist_ok=True)

        self.PCC_ref = self.script_dir/"FGGGG.pdb" # path to refrence PCC structure

        if target == "FEN": # path to Fentanyl .itp and .pdb files
            self.target_dir = Path(self.settings['FEN_dir'])
        elif target == "DEC":# path to decoy .itp and .pdb files
            self.target_dir = Path(self.settings['DEC_dir'])
        else:
            raise ValueError(f"Target {target} is not defined. Select 'FEN' for Fentanyl or 'DEC' for Benzyl-Fentanyl.")
        self.target = target
        
        self.AAdict31 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'} # 3 to 1 translator
        self.AAdict13 = {j: i for i, j in self.AAdict31.items()} # 1 to 3 translator
        self.charge_dict = {"D": -1, "E": -1, "R": +1, "K": +1} # AA charges at neutral pH

        self.PCC_charge = sum([self.charge_dict.get(i, 0) for i in list(self.PCC_code)]) # Net charge of the PCC at pH=7
        self.PCC_n_atoms = None # number of PCC atoms
        self.MOL_list = [] # list of MOL atom ids (str)
        self.PCC_list = [] # list of PCC atom ids (str)
        self.MOL_list_atom = [] # list of MOL atom names (str)
        self.PCC_list_atom = [] # list of PCC atom names (str)
        self.free_e = None # Free energy of binding kJ/mol
        self.free_e_err = None # Error in free energy of binding kJ/mol
        self.K = None # Binding constant
        self.K_err = None # Error in binding constant
        self.KbT = float(self.settings["T"]) * 8.314 # System temperature in J/mol

        self.pymol = Path(self.settings['pymol_dir']) # path to pymol installation
        self.packmol = Path(self.settings['packmol_dir']) # path to packmol installation
        self.acpype = Path(self.settings['acpype_dir']) # Path to acpype conda env
    
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
                atom_coordinates.append(np.array([float(line_list[6]), float(line_list[7]), float(line_list[8])]))
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
    
    def _write_report(self):
        report = {
            "PCC": self.PCC_code,
            "Target": self.target,
            "FE": self.free_e,
            "FE_error": self.free_e_err,
            "K": self.K,
            "K_err": self.K_err
        }
        with open(self.base_dir/"metadata.JSON", 'w') as f:
            json.dump(report, f, indent=3)
        return None
    
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
        self._set_done(self.PCC_dir) # mark stage as done
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

    def _prep_pdb(self) -> None:
        """
        acpype only accepts pdb files with one residue. Remove all residue information from acpype input file.

        Returns:
            None
        """
        with cd(self.PCC_dir): # cd into the PCC directory
            with open(f"{self.PCC_code}.pdb") as f:
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
        # create acpype pdb with 1 residue
        self._prep_pdb()

        wait_str = " --wait " if wait else "" # whether to wait for acpype to finish before exiting
        with cd(self.PCC_dir): # cd into the PCC directory and run acpype.
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_acpype.sh {self.PCC_code}_acpype {self.PCC_charge}", shell=True, check=True)
        
        self._set_done(self.PCC_dir/"PCC.acpype")
        return None
    
    def _minimize_PCC(self, wait: bool = True) -> None: 
        """
        Run minimization for PCC. Copies acpype files into `em` directory, solvates, adds ions, and minimizes
        the structure. The last frame is save as `PCC.gro`

        Args:
            wait (bool, optional): Whether to wait for em to finish. Defaults to True.

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
            # get last frame of em (NOT NECESSARY)
            # subprocess.run(f"bash {self.script_dir}/get_last_frame.sh -f ./em.trr -s ./em.tpr -o ./PCC_em.pdb", check=True, shell=True)
        self._set_done(self.PCC_dir/"em")

        return None
    
    def _get_atom_ids(self, gro_file: Path) -> None:
        """
        Get the atom ids after mixing. Automatically called after em step on the complex.

        Args:
            gro_file (Path): path to the gro file

        Raises:
            FileNotFoundError: Raised if `gro_file` is not found.

        Returns:
            None
        """
        # check if file exists
        if not Path(gro_file).exists():
            raise FileNotFoundError("gro not in the working directory.")
        # read gro
        with open(gro_file) as f:
            em_cnt = f.readlines()
        # get atom list
        atom_list = em_cnt[2:-1]
        # define atom id lists
        MOL_list_id = []
        MOL_list_atom = []
        PCC_list_id = []
        PCC_list_atom = []
        # get atom ids
        for line in atom_list:
            line_list = line.split()
            if re.match("^\d+MOL$", line_list[0]):
                MOL_list_id.append(int(line_list[2]))
                MOL_list_atom.append(line_list[1])
            elif re.match("^\d+PCC$", line_list[0]):
                PCC_list_id.append(int(line_list[2]))
                PCC_list_atom.append(line_list[1])
        # save MOL_list and PCC_list
        self.MOL_list = MOL_list_id
        self.PCC_list = PCC_list_id
        self.MOL_list_atom = MOL_list_atom
        self.PCC_list_atom = PCC_list_atom
        return None
        
    def _fix_posre(self) -> None:
        """
        Fix atom ids in position restraint files. Read the new ids from the `em.gro` file
        AFTER minimization and writes posre_MOL.itp and posre_PCC.itp.
        atoms.

        Returns:
            None
        """
        # write posre_MOL.itp
        cwd = os.getcwd()
        if Path("./posre_MOL.itp").exists():
            subprocess.run(f"mv {cwd}/posre_MOL.itp {cwd}/posre_MOL_backup.itp", shell=True)
        with open("./posre_MOL.itp", 'w') as f:
            f.write("; posre_MOL.itp\n")
            f.write("\n")
            f.write("[ position_restraints ]\n")
            f.write("; atom  type      fx      fy      fz\n")
            for i in self.MOL_list:
                f.write(f"\t{i}\t1\t1000\t1000\t1000\n")
        # write posre_PCC.itp
        if Path("./posre_PCC.itp").exists():
            subprocess.run(f"mv {cwd}/posre_PCC.itp {cwd}/posre_PCC_backup.itp", shell=True)
        with open("./posre_PCC.itp", 'w') as f:
            f.write("; posre_PCC.itp\n")
            f.write("\n")
            f.write("[ position_restraints ]\n")
            f.write("; atom  type      fx      fy      fz\n")
            for i in self.PCC_list:
                f.write(f"\t{i}\t1\t1000\t1000\t1000\n")
        return None

    def _mix(self) -> None:
        """
        Create the simulation box with MOL and PCC, and create initial structures for both sides of the PCC.

        Returns:
            None
        """
        if not self._check_done(self.complex_dir):
        ## CREATE TOPOL.TOP
            with cd(self.complex_dir): # cd into complex
                # copy MOL and PCC files into complex directory
                subprocess.run(f"cp {self.target_dir}/MOL.itp .", shell=True, check=True)
                subprocess.run(f"cp {self.target_dir}/MOL.pdb .", shell=True, check=True)
                subprocess.run(f"cp {self.target_dir}/posre_MOL.itp .", shell=True, check=True) # This has incorrect atom numbers
                subprocess.run(f"cp {self.PCC_dir}/PCC.acpype/PCC_GMX.itp ./PCC.itp", shell=True, check=True)
                subprocess.run(f"cp {self.PCC_dir}/PCC.acpype/posre_PCC.itp .", shell=True, check=True)
                subprocess.run(f"cp {self.PCC_dir}/em/PCC_em.pdb ./PCC.pdb", shell=True, check=True)
                # create complex.pdb with packmol
                subprocess.run(f"cp {self.mold_dir}/complex/mix/mix.inp .", shell=True, check=True)
                subprocess.run(f"module unload gcc && module load gcc/10.2.0 && {self.packmol} < ./mix.inp", shell=True, check=True)
                # create topol.top and complex.itp
                top = GMXitp("./MOL.itp", "./PCC.itp")
                top.create_topol()

            self._set_done(self.complex_dir)
        
        return None
    
    def _eq_complex(self, wait: bool = True) -> None:
        """
        Solvate, and equilibrate the complex.

        Args:
            rot_key: (int): Which side of the PCC to equilibrate
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

        Returns:
            None
        """
        ## EM
        if not self._check_done(self.complex_dir/"em"):
            # create complex/em dir
            Path.mkdir(self.complex_dir/"em", exist_ok=True)
            with cd(self.complex_dir/"em"): # cd into complex/em
                # copy files into complex/em
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run(f"cp ../complex.pdb .", shell=True, check=True)
                subprocess.run("cp ../topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/em/ions.mdp .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/em/em.mdp .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/em/sub_mdrun_complex_em.sh .", shell=True) # copy mdrun submission script
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_em.sh {self.PCC_charge}", check=True, shell=True)
            self._set_done(self.complex_dir/'em')
        with cd(self.complex_dir/"em"): # cd into complex/em
            # update atom ids
            self._get_atom_ids("./em.gro")
        ## NVT
        if not self._check_done(self.complex_dir/"nvt"):
            # create complex/nvt dir
            Path.mkdir(self.complex_dir/"nvt", exist_ok=True)
            with cd(self.complex_dir/"nvt"): # cd into complex/nvt
                # copy files into complex/nvt
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../em/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/nvt/sub_mdrun_complex_nvt.sh .", shell=True) # copy mdrun submission script
                # copy nvt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt_nions.mdp ./nvt.mdp", shell=True, check=True)
                # submit nvt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_nvt.sh", check=True, shell=True)
            self._set_done(self.complex_dir/'nvt')
        ## NPT
        if not self._check_done(self.complex_dir/"npt"):
            # create complex/npt dir
            Path.mkdir(self.complex_dir/"npt", exist_ok=True)
            with cd(self.complex_dir/"npt"): # cd into complex/npt
                # copy files into complex/npt
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../nvt/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/npt/sub_mdrun_complex_npt.sh .", shell=True) # copy mdrun submission script
                # copy npt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.mold_dir}/complex/npt/npt.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/npt/npt_nions.mdp ./npt.mdp", shell=True, check=True)
                # submit npt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_npt.sh", shell=True)
            self._set_done(self.complex_dir/'npt')

    def _is_continuous(self, ids: list) -> bool:
        """
        Check if `ids` contains a continuous list of atom ids.

        Args:
            ids (list): list of atom ids

        Returns:
            bool: True if is continuous, False otherwise
        """
        prev_id = None
        for id in ids:
            if prev_id is None:
                prev_id = id
                continue
            if id-prev_id == 1:
                prev_id = id
            else:
                return False
        return True

    def _create_plumed(self, plumed_in: Path, plumed_out: Path) -> None:
        """
        Fix the plumed tempelate with MOL and PCC atom ids. DOES NOT SUPPORT NON-CONTINUOUS ATOM IDS.

        Args:
            plumed_in (Path): Path to input plumed file
            plumed_out (Path): Path to output plumed file

        Raises:
            AssertionError: If `self.MOL_list` or `self.PCC_list` are not continuous.
            
        Returns:
            None
        """
        # read plumed files
        with open(plumed_in) as f:
            cnt = f.readlines()
        # make sure id lists are continuous
        assert self._is_continuous(self.MOL_list), "MOL id list is not continuous. Check complex/em/em.gro."
        assert self._is_continuous(self.PCC_list), "PCC id list is not continuous. Check complex/em/em.gro."
        # define atom ranges for PCC and MOL
        MOL_atom_id = f"{min(self.MOL_list)}-{max(self.MOL_list)}"
        PCC_atom_id = f"{min(self.PCC_list)}-{max(self.PCC_list)}"
        a_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in ["N4", "C10", "C11", "O1"]]
        b_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in ["C1", "C2", "C3", "O"]]
        v1a_atom_ids = "".join([f"{i}," for i in a_list])[:-1]
        v1b_atom_ids = "".join([f"{i}," for i in b_list])[:-1]
        b_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in ["N1", "N2", "N3", "C7", "C8"]]
        vrb_atom_ids = "".join([f"{i}," for i in b_list])[:-1]
        if self.target == "FEN":
            a_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in ["C", "C1", "C2"]]
            b_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in ["N1", "C13", "C21"]]
        else:
            a_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in ["C", "C1", "N"]]
            b_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in [f"C{j}" for j in range(2, 8)]]
        
        v2a_atom_ids = "".join([f"{i}," for i in a_list])[:-1]
        v2b_atom_ids = "".join([f"{i}," for i in b_list])[:-1]

        # replace new ids
        for i, line in enumerate(cnt):
            if "$1" in line:
                line = line.replace("$1", PCC_atom_id)
            if "$2" in line:
                line = line.replace("$2", MOL_atom_id)
            if "$3" in line:
                line = line.replace("$3", v1a_atom_ids)
            if "$4" in line:
                line = line.replace("$4", v1b_atom_ids)
            if "$5" in line:
                line = line.replace("$5", v2a_atom_ids)
            if "$6" in line:
                line = line.replace("$6", v2b_atom_ids)
            if "$7" in line:
                line = line.replace("$7", vrb_atom_ids)
            
            cnt[i] = line
        # write new plumed file
        with open(plumed_out, 'w') as f:
            f.writelines(cnt)

        return None

    def _pbmetaD(self, wait: bool = True) -> None:
        """
        Run PBMetaD from equilibrated structure.

        Args:
            rot_key: (int): Which side of the PCC to run.
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

        Returns:
            None
        """
        # create complex/pbmetad dir
        Path.mkdir(self.complex_dir/"md", exist_ok=True)
        with cd(self.complex_dir/"md"): # cd into complex/pbmetad
            wait_str = " --wait " if wait else "" # whether to wait for pbmetad to finish before exiting
            if Path.exists(self.complex_dir/"md"/"md.cpt"):
                now = datetime.now()
                now = now.strftime("%m/%d/%Y, %H:%M:%S")
                print(f"{now}: Resuming previous run...")
            else:
                # copy files into complex/pbmetad
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../npt/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.mold_dir}/complex/md/sub_mdrun_plumed.sh .", shell=True) # copy mdrun submission script
                subprocess.run(f"cp {self.mold_dir}/complex/md/plumed.dat ./plumed_temp.dat", shell=True) # copy pbmetad script
                subprocess.run(f"cp {self.mold_dir}/complex/md/plumed_restart.dat ./plumed_r_temp.dat", shell=True) # copy pbmetad script
                # update PCC and MOL atom ids
                self._create_plumed("./plumed_temp.dat", "./plumed.dat")
                self._create_plumed("./plumed_r_temp.dat", "./plumed_restart.dat")
                # remove temp plumed file
                subprocess.run(f"rm ./plumed_temp.dat", shell=True)
                subprocess.run(f"rm ./plumed_r_temp.dat", shell=True)
                # copy nvt.mdp into pbmetad
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.mold_dir}/complex/md/md.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/md/md_nions.mdp ./md.mdp", shell=True, check=True)
            # submit pbmetad job. Resubmits until either the job fails 10 times or it succesfully finishes.
            cnt = 1
            try:
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
            except:
                fail_flag = True
                while fail_flag:
                    try:
                        cnt += 1
                        subprocess.run(f"mv ./HILLS_ang ./HILLS_ang.bck.{cnt}", shell=True, check=True)
                        subprocess.run(f"mv ./HILLS_cos ./HILLS_cos.bck.{cnt}", shell=True, check=True)
                        subprocess.run(f"mv ./HILLS_COM ./HILLS_COM.bck.{cnt}", shell=True, check=True)
                        now = datetime.now()
                        now = now.strftime("%m/%d/%Y, %H:%M:%S")
                        print(f"{now}: Resubmitting PBMetaD: ", end="", flush=True)
                        subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
                        print()
                        fail_flag = False
                    except:
                        if cnt >= 10:
                            raise RuntimeError("PBMetaD run failed more than 10 times. Stopping.")
    
        self._set_done(self.complex_dir/'md')
        return None
    
    def _reweight(self, wait: bool = True) -> None:
        """
        Reweight the results of the pbmetad run.

        Args:
            rot_key: (int): Which side of the PCC to run.
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

        Returns:
            None
        """
        # create complex/reweight dir
        Path.mkdir(self.complex_dir/"reweight", exist_ok=True)
        with cd(self.complex_dir/"reweight"): # cd into complex/reweight
            # copy files into complex/reweight
            #subprocess.run("cp ../pbmetad/HILLS_COM .", shell=True, check=True)
            #subprocess.run("cp ../pbmetad/HILLS_ang .", shell=True, check=True)
            subprocess.run("cp ../md/GRID_COM .", shell=True, check=True)
            subprocess.run("cp ../md/GRID_ang .", shell=True, check=True)
            subprocess.run("cp ../md/GRID_cos .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/complex/reweight/sub_mdrun_rerun.sh .", shell=True) # copy mdrun submission script            
            subprocess.run(f"cp {self.mold_dir}/complex/reweight/reweight.dat ./reweight_temp.dat", shell=True) # copy reweight script
            # update PCC and MOL atom ids
            self._create_plumed("./reweight_temp.dat", "./reweight.dat")
            # remove temp plumed file
            subprocess.run(f"rm ./reweight_temp.dat", shell=True)
            # submit reweight job
            wait_str = " --wait " if wait else "" # whether to wait for reweight to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_rerun.sh", check=True, shell=True)
        self._set_done(self.complex_dir/'reweight')
        return None
    
    def _block_anal(self, x: list, weights: list, S_cor: bool = False, folds: int = 5):
        """
        Block analysis for a collective variable.

        Args:
            x (list): List of colvar values
            weights (list): List of colvar weights
            S_cor (bool, optional): Whether to include entropy corrections
            folds (int, optional): number of blocks for block analysis

        Returns:
            data_s (dict): Dictionary with free energy of all folds, free energy from all data, and standard errors from block analysis
        """
        _, bins = np.histogram(x, bins=100)
        xs = (bins[1:] + bins[:-1])/2
        block_size = len(x)//folds
        data = pd.DataFrame()
        data['bin_center'] = xs
        for fold in range(folds):
            counts, _ = np.histogram(x[block_size*fold:(fold+1)*block_size], bins=bins, weights=weights[block_size*fold:(fold+1)*block_size])
            Fs = -self.KbT*np.log(counts)/1000 #kJ/mol
            if S_cor:
                Fs += 2*self.KbT*np.log(xs)/1000
            data[f"f_{fold}"] = Fs

        data_s = pd.DataFrame()
        data_s['bin_center'] = data['bin_center']
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        for fold in range(folds):
            data_s[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"].mean()

        data_s_temp = data_s.replace([np.inf, -np.inf], np.nan)
        data_s['std'] = data_s_temp.apply(np.std, axis=1)
        data_s['ste'] = 1/np.sqrt(folds)*data_s['std']

        counts, _ = np.histogram(x, bins=bins, weights=weights)
        Fs = -self.KbT*np.log(counts)/1000 #kJ/mol
        if S_cor:
            Fs += 2*self.KbT*np.log(xs)/1000

        data_s["f_all"] = Fs
        data_s["f_all"] = data_s["f_all"] - data_s["f_all"].mean()
        return data_s

    def _find_converged(self):
        """
        Find the simulation time from which point the simulations can be considered converged.
        This is used to discard the initial stage of the simulation where the bias value is <85% of the maximum.
        """
        with open(self.complex_dir/"md"/"COLVAR", 'r') as f:
            fields = f.readline()[:-1].split(" ")[2:] # last char is '/n' and the first two are '#!' and 'FIELDS'
            time_ind = fields.index("time")
            bias_ind = fields.index("pb.bias")
            bias = []
            line = f.readline()
            while line: # read up to LINE_LIM lines
                if line[0] == "#": # Don't read comments
                    line = f.readline()
                    continue
                line_list = line.split()
                bias.append([float(line_list[time_ind]), float(line_list[bias_ind])])
                line = f.readline()
            bias = np.asarray(bias)
            bias = bias[bias[:, 1] > 0.8*bias[:, 1].max()]
            return bias[0, 0]//1000
    
    def _load_plumed(self):
        data = {}
        with open(self.complex_dir/"reweight"/"COLVAR", 'r') as f:
            fields = f.readline()[:-1].split(" ")[2:] # last char is '/n' and the first two are '#!' and 'FIELDS'
            for field in fields: # add fields to the colvars dict
                data[field] = []
            line = f.readline()
            while line: # read up to LINE_LIM lines
                if line[0] == "#": # Don't read comments
                    line = f.readline()
                    continue
                line_list = line.split()
                for i, field in enumerate(fields):
                    data[field].append(float(line_list[i]))
                line = f.readline()
        data = pd.DataFrame(data)
        data['weights'] = np.exp(data['pb.bias']*1000/self.KbT)
        init_time = self._find_converged() #ns
        print(f"INFO: Discarding initial {init_time} ns of data for free energy calculations.")
        if init_time > 300:
            raise RuntimeError("Large hill depositions detected past 300 ns mark. Check the convergence of the PBMetaD calculations.")
        init_idx = int(init_time * 10000 // 2)
        data = data.iloc[init_idx:] # discard the first 100 ns of data
        return data
    
    def _block_anal_2d(self, x, y, weights, block_size=None, folds=None, nbins=100):
        # calculate histogram for all data to get bins
        _, binsx, binsy = np.histogram2d(x, y, bins=nbins, weights=weights)
        # calculate bin centers
        xs = np.round((binsx[1:] + binsx[:-1])/2, 2)
        ys = np.round((binsy[1:] + binsy[:-1])/2, 2)
        # find block sizes
        if block_size is None:
            if folds is None:
                block_size = 5000*50 #50 ns blocks
                folds = len(x)//block_size
            else:
                block_size = len(x)//folds
        else:
            folds = len(x)//block_size

        # data frame to store the blocks
        data = pd.DataFrame()
        xs_unrolled = []
        ys_unrolled = []
        for i in xs:
            for j in ys:
                xs_unrolled.append(i)
                ys_unrolled.append(j)
        
        data['x'] = xs_unrolled
        data['y'] = ys_unrolled

        # calculate free energy for each fold
        for fold in range(folds):
            x_fold = x[block_size*fold:(fold+1)*block_size]
            y_fold = y[block_size*fold:(fold+1)*block_size]
            weights_fold = weights[block_size*fold:(fold+1)*block_size]
            counts, _, _ = np.histogram2d(x_fold, y_fold, bins=[binsx, binsy], weights=weights_fold)
            counts[counts==0] = np.nan # discard empty bins
            free_energy = -self.KbT*np.log(counts)/1000 #kJ/mol
            free_energy_unrolled = []
            for i in range(len(xs)):
                for j in range(len(ys)):
                    free_energy_unrolled.append(free_energy[i, j])
            data[f"f_{fold}"] = free_energy_unrolled
            # Entropy correction along x axis
            data[f"f_{fold}"] += 2*self.KbT*np.log(data.x)/1000
        
        # de-mean the folds for curve matching
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        for fold in range(folds):
            data[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"].mean()
        
        # calcualte standard deviation and standard error
        data['std'] = data[[f"f_{fold}" for fold in range(folds)]].apply(np.std, axis=1)
        data['ste'] = 1/np.sqrt(folds)*data['std']

        return data
    
    def _calc_region_int(self, data):
        """
        data = DataFrame with columns x(dcom), y(angle), F(free energy)
        """
        data["exp"] = np.exp(-data.F*1000/self.KbT)
        Y_integrand = {"X": [], "exp":[]}
        # integrate over Y
        for x in data.x.unique():
            FE_this_x = data[data.x == x]
            Y_integrand["X"].append(x)
            Y_integrand["exp"].append(simpson(y=FE_this_x.exp.to_numpy(), x=FE_this_x.y.to_numpy()))
        
        Y_integrand_pd = pd.DataFrame(Y_integrand)
        # integrate over X
        integrand = simpson(y=Y_integrand_pd.exp.to_numpy(), x=Y_integrand_pd.X.to_numpy())
        
        return -self.KbT*np.log(integrand)/1000

    def _calc_deltaF(self, bound_data, unbound_data):
        r_int = self._calc_region_int(bound_data.copy())
        p_int = self._calc_region_int(unbound_data.copy())
        return r_int - p_int
    
    def _calc_FE(self) -> None:
        colvars = self._load_plumed() # read colvars
        # block analysis
        block_anal_data = self._block_anal_2d(colvars.dcom, colvars.ang,
                                        colvars.weights, nbins=50, block_size=5000*100)
        f_list = []
        f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
        for i in f_cols:
            # bound = 0<=dcom<=1.5 nm
            bound_data = block_anal_data[(block_anal_data.x>=0.0) & (block_anal_data.x<=1.5)][['x', 'y', i, 'ste']]
            bound_data.rename(columns={i: 'F'}, inplace=True)
            bound_data.dropna(inplace=True)
            # unbound = 2.0<dcom<2.4~inf nm 
            unbound_data = block_anal_data[(block_anal_data.x>2.0) & (block_anal_data.x<2.4)][['x', 'y', i, 'ste']]
            unbound_data.rename(columns={i: 'F'}, inplace=True)
            unbound_data.dropna(inplace=True)
            f_list.append(self._calc_deltaF(bound_data=bound_data, unbound_data=unbound_data))
        f_list = np.array(f_list)
        return np.nanmean(f_list), np.nanstd(f_list)/np.sqrt(len(f_list)-np.count_nonzero(np.isnan(f_list)))
    
    def _calc_K(self) -> tuple:
        self.K = np.exp(-self.free_e*1000/self.KbT)
        self.K_err = self.K*self.free_e_err*1000/self.KbT
        return self.K, self.K_err
    
    def _postprocess(self) -> None:
        # calc FE
        if not (self.base_dir/"metadata.JSON").exists():
            free_e, free_e_err = self._calc_FE()
            self.free_e = free_e
            self.free_e_err = free_e_err
            # calculate Ks
            self._calc_K()
            # write report
            self._write_report()
        return None

    
    def run(self) -> float:
        """Wrapper for FE caclculations. Create PCC, call acpype, 
        minimize, create complex, and run PBMetaD.

        Returns:
            self.free_e (dict): Free energy of adsorption b/w MOL and PCC
            self.free_e_err (dict): Error in free energy of adsorption b/w MOL and PCC
        """
        # create PCC
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Free energy calculations for {self.PCC_code} with {self.target}")
        print(f"{now}: Running pymol: ", end="", flush=True)
        if not self._check_done(self.PCC_dir):
            self._create_pcc()
        print("\tDone.", flush=True)
        # get params
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Running acpype: ", end="", flush=True)
        if not self._check_done(self.PCC_dir/"PCC.acpype"):
            self._get_params()
        print("\tDone.", flush=True)
        # minimize PCC
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Minimizing PCC: ", end="", flush=True)
        if not self._check_done(self.PCC_dir/'em'):
            self._minimize_PCC()
        print("\tDone.", flush=True)
        # create the complex
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Creating complex box: ", end="", flush=True)
        if not self._check_done(self.complex_dir):
            self._mix()
        print("\tDone.", flush=True)
        # minimize the complex
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Minimizing and equilibrating the complex box: ", end="", flush=True)
        self._eq_complex()
        print("\tDone.", flush=True)
        # run PBMetaD
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Running PBMetaD: ", end="", flush=True)
        if not self._check_done(self.complex_dir/'md'):
            self._pbmetaD()
        print("\tDone.")
        # reweight
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Reweighting PBMetaD results: ", end="", flush=True)
        if not self._check_done(self.complex_dir/'reweight'):
            self._reweight()
        print("\tDone.", flush=True)
        #postprocess
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Postprocessing: ")
        self._postprocess()
        print(f"{now}: All steps completed.")
        print("-"*30 + "Finished" + "-"*30)
        
        return self.free_e, self.free_e_err
