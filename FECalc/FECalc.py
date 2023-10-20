import os
import re
import subprocess
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import simpson

from .GMXitp import GMXitp

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
        self.script_dir = Path(self.settings['scripts_dir'])
        self.PCC_code = pcc
        base_dir = Path(base_dir)
        if base_dir.exists():
            if not base_dir.is_dir():
                raise ValueError(f"{base_dir} is not a directory.")
        else:
            print("Base directory does not exist. Creating...")
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
        self.free_e = None # Free energy of binding kJ/mol
        self.free_e_err = None # Error in free energy of binding kJ/mol
        self.KbT = float(self.settings["T"]) * 8.314 # System temperature in J/mol

        self.pymol = Path(self.settings['pymol_dir']) # path to pymol installation
        self.packmol = Path(self.settings['packmol_dir']) # path to packmol installation
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
        subprocess.run(f"cp {self.script_dir}/sub_acpype.sh {self.PCC_dir}", shell=True) # copy acpype submission script
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
            subprocess.run(f"cp {self.script_dir}/topol.top .", shell=True, check=True)
            subprocess.run(f"cp {self.script_dir}/ions.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.script_dir}/em.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.script_dir}/sub_mdrun_PCC_em.sh .", shell=True) # copy mdrun submission script
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
        MOL_list = []
        PCC_list = []
        # get atom ids
        for line in atom_list:
            line_list = line.split()
            if re.match("^\d+MOL$", line_list[0]):
                MOL_list.append(int(line_list[2]))
            elif re.match("^\d+PCC$", line_list[0]):
                PCC_list.append(int(line_list[2]))
        # save MOL_list and PCC_list
        self.MOL_list = MOL_list
        self.PCC_list = PCC_list
        return None
        
    def _fix_posre(self) -> None: # NOTE: Not tested
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

    def _mix(self, wait: bool = True) -> None:
        """
        Create the simulation box with MOL and PCC, solvate, and equilibrate.

        Args:
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

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
                subprocess.run(f"cp {self.script_dir}/mix.inp .", shell=True, check=True)
                subprocess.run(f"module unload gcc && module load gcc/10.2.0 && {self.packmol} < ./mix.inp", shell=True, check=True)
                # create topol.top and complex.itp
                top = GMXitp("./MOL.itp", "./PCC.itp")
                top.create_topol()
            self._set_done(self.complex_dir)
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
                subprocess.run("cp ../complex.pdb .", shell=True, check=True)
                subprocess.run("cp ../topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/ions.mdp .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/em.mdp .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/sub_mdrun_complex_em.sh .", shell=True) # copy mdrun submission script
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_em.sh {self.PCC_charge}", check=True, shell=True)
            self._set_done(self.complex_dir/'em')
        with cd(self.complex_dir/"em"): # cd into complex/em
            # update atom ids
            self._get_atom_ids("./em.gro")
        if not self._check_done(self.complex_dir/"nvt"):
            ## NVT
            # create complex/nvt dir
            Path.mkdir(self.complex_dir/"nvt", exist_ok=True)
            with cd(self.complex_dir/"nvt"): # cd into complex/nvt
                # copy files into complex/nvt
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../em/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/sub_mdrun_complex_nvt.sh .", shell=True) # copy mdrun submission script
                # copy nvt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.script_dir}/nvt.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.script_dir}/nvt_noions.mdp ./nvt.mdp", shell=True, check=True)
                # submit nvt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_nvt.sh", check=True, shell=True)
            self._set_done(self.complex_dir/'nvt')
        if not self._check_done(self.complex_dir/"npt"):
            ## NPT
            # create complex/npt dir
            Path.mkdir(self.complex_dir/"npt", exist_ok=True)
            with cd(self.complex_dir/"npt"): # cd into complex/npt
                # copy files into complex/npt
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../nvt/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/sub_mdrun_complex_npt.sh .", shell=True) # copy mdrun submission script
                # copy npt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.script_dir}/npt.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.script_dir}/npt_noions.mdp ./npt.mdp", shell=True, check=True)
                # submit npt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_npt.sh npt", shell=True)
            self._set_done(self.complex_dir/'npt')
        return None
    
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
        # replace new ids
        for i, line in enumerate(cnt):
            if "$1" in line:
                line = line.replace("$1", MOL_atom_id)
            if "$2" in line:
                line = line.replace("$2", PCC_atom_id)
            cnt[i] = line
        # write new plumed file
        with open(plumed_out, 'w') as f:
            f.writelines(cnt)

        return None

    def _pbmetaD(self, wait: bool = True) -> None:
        """
        Run PBMetaD from equilibrated structure.

        Args:
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

        Returns:
            None
        """
        # create complex/pbmetad dir
        Path.mkdir(self.complex_dir/"pbmetad", exist_ok=True)
        with cd(self.complex_dir/"pbmetad"): # cd into complex/pbmetad
            wait_str = " --wait " if wait else "" # whether to wait for pbmetad to finish before exiting
            if Path.exists(self.complex_dir/"pbmetad"/"md.cpt"):
                print("Resuming previous run...")
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed_restart.sh", check=True, shell=True)
            else:
                # copy files into complex/pbmetad
                subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_MOL.itp .", shell=True, check=True)
                subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
                subprocess.run("cp ../em/posre_PCC.itp .", shell=True, check=True)
                subprocess.run("cp ../complex.itp .", shell=True, check=True)
                subprocess.run("cp ../npt/topol.top .", shell=True, check=True)
                subprocess.run(f"cp {self.script_dir}/sub_mdrun_plumed.sh .", shell=True) # copy mdrun submission script
                subprocess.run(f"cp {self.script_dir}/sub_mdrun_plumed_restart.sh .", shell=True) # copy restart submission script
                subprocess.run(f"cp {self.script_dir}/plumed.dat ./plumed_temp.dat", shell=True) # copy pbmetad script
                subprocess.run(f"cp {self.script_dir}/plumed_restart.dat ./plumed_r_temp.dat", shell=True) # copy pbmetad script
                # update PCC and MOL atom ids
                self._create_plumed("./plumed_temp.dat", "./plumed.dat")
                self._create_plumed("./plumed_r_temp.dat", "./plumed_restart.dat")
                # remove temp plumed file
                subprocess.run(f"rm ./plumed_temp.dat", shell=True)
                subprocess.run(f"rm ./plumed_r_temp.dat", shell=True)
                # copy nvt.mdp into pbmetad
                if self.PCC_charge != 0:
                    subprocess.run(f"cp {self.script_dir}/md.mdp .", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.script_dir}/md_noions.mdp ./md.mdp", shell=True, check=True)
                # submit pbmetad job
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed_restart.sh", check=True, shell=True)
        self._set_done(self.complex_dir/'pbmetad')
        return None
    
    def _reweight(self, wait: bool = True) -> None:
        """
        Reweight the results of the pbmetad run.

        Args:
            wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

        Returns:
            None
        """
        # create complex/reweight dir
        Path.mkdir(self.complex_dir/"reweight", exist_ok=True)
        with cd(self.complex_dir/"reweight"): # cd into complex/reweight
            # copy files into complex/reweight
            subprocess.run("cp ../pbmetad/HILLS_COM .", shell=True, check=True)
            subprocess.run("cp ../pbmetad/HILLS_MOL .", shell=True, check=True)
            subprocess.run("cp ../pbmetad/HILLS_PCC .", shell=True, check=True)
            subprocess.run("cp ../pbmetad/COLVAR .", shell=True, check=True)
            subprocess.run(f"cp {self.script_dir}/sub_mdrun_rerun.sh .", shell=True) # copy mdrun submission script            
            subprocess.run(f"cp {self.script_dir}/reweight.dat ./reweight_temp.dat", shell=True) # copy reweight script
            # update PCC and MOL atom ids
            self._create_plumed("./reweight_temp.dat", "./reweight.dat")
            # remove temp plumed file
            subprocess.run(f"rm ./reweight_temp.dat", shell=True)
            # submit reweight job
            wait_str = " --wait " if wait else "" # whether to wait for reweight to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_rerun.sh", check=True, shell=True)
        self._set_done(self.complex_dir/'reweight')
        return None
    
    def _load_plumed(self, fname: str, line_lim: int = None):
        """
        Helper function to read PLUMED output files.

        Args:
            fname (str): name of the plumed output file
            line_lim(int, optional): number of lines to read
        
        Returns:
            data (dict[str: list]): dictionary containing `field`: values
        """
        data = {}
        with open(fname, 'r') as f:
            fields = f.readline()[:-1].split(" ")[2:] # last char is '/n' and the first two are '#!' and 'FIELDS'
            for field in fields: # add fields to the colvars dict
                data[field] = []
            line_cnt = 0
            cond = True
            while cond: # read up to LINE_LIM lines
                line = f.readline()
                if not line:
                    break
                if line[0] == "#": # Don't read comments
                    continue
                
                line_list = line.split()
                try: # account for incomplete files (useful when the sims is still running)
                    for i, field in enumerate(fields):
                        data[field].append(float(line_list[i]))
                except:
                    break
                finally:
                    line_cnt += 1
                    if line_lim is not None:
                        cond = (line_cnt<line_lim)
        return data
    
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

    def _find_flat(self, y, x, max_df = 5):
        """
        Find the section of the PMF that is flat enough to be considered constant.

        Args:
        y (ListLike): Free energy list
        x (ListLike): Colvar list
        max_df (float, optional): maximum allowable gradient

        Returns:
        ind (int): Starting index of the flat section\
        
        Raises:
        ValueError: When no flat section is found.
        """
        dx = np.gradient(y, x)
        for ind in range(len(dx)):
            if np.mean(np.abs(dx[ind:])) < max_df and (ind < len(dx) - 10):
                return ind
        raise ValueError("No stretch of the free energy profile is flat enough. Maybe PBMETAD is not converged?")

    def _calc_FE(self) -> None:
        # find the timestep where the hamiltonian is pseudo-constant
        with cd(self.complex_dir/"pbmetad"): # cd into pbmetad directory
            hills = self._load_plumed("HILLS_COM")
        
        for start_t in (range(len(hills["height"])//10000)):
            if max(hills["height"][start_t*10000:]) < 0.04:
                start_t = start_t*10
                break
        # get the free energy profile w.r.t. com-com distance
        with cd(self.complex_dir/"reweight"): # cd into reweight directory
            colvars = self._load_plumed("COLVAR_RW")
        colvars = pd.DataFrame(colvars)
        colvars['weights'] = np.exp(colvars['pb.bias']*1000/self.KbT) # Calculate reweighting weights
        colvars = colvars.iloc[start_t * 10000 // 2:] # discard all data before pseudo-constant hamiltonian section
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        FE_data = self._block_anal(colvars.dcom, colvars.weights, S_cor=True, folds=5) # block analysis to get FE and errors
        # find the adsorption FE
        # discard data from bins > 3.5 nm
        FE_data = FE_data[FE_data.bin_center < 3.5]
        # Attached
        bin_widths = FE_data.bin_center.to_numpy().copy()[1:]
        bin_widths -= FE_data.bin_center.to_numpy()[:-1]
        FE_well_df = FE_data[FE_data.bin_center < 1.5]
        FE_well = FE_well_df.f_all.to_numpy()
        FE_well_int = simpson(y=np.exp(-FE_well*1000/self.KbT), x=FE_well_df.bin_center.to_numpy())
        FE_well_val = -self.KbT*np.log(FE_well_int)/1000
        FE_well_err = FE_data[FE_data.bin_center < 1.5].ste.to_numpy()
        FE_well_sum = np.sum(np.exp(-FE_well*1000/self.KbT))
        FE_well_err = np.exp(-FE_well*1000/self.KbT)/FE_well_sum * FE_well_err
        FE_well_err = np.linalg.norm(FE_well_err)
        # Free
        flat_sec_s_ind = self._find_flat(FE_data.f_all, FE_data.bin_center)
        FE_free_val = FE_data.f_all[flat_sec_s_ind:].mean()
        FE_free_err = FE_data.ste[flat_sec_s_ind:].to_numpy()
        FE_free_err = np.linalg.norm(FE_free_err)/len(FE_free_err)
        # diff
        FE_bind_val = FE_well_val - FE_free_val
        FE_bind_err = np.sqrt(FE_well_err**2 + FE_free_err**2)

        return FE_bind_val, FE_bind_err
    
    def get_FE(self) -> float:
        """Wrapper for FE caclculations. Create PCC, call acpype, 
        minimize, create complex, and run PBMetaD.

        Returns:
            self.free_e (float): Free energy of adsorption b/w MOL and PCC
        """
        # create PCC
        print("Running pymol:", end="", flush=True)
        if not self._check_done(self.PCC_dir):
            self._create_pcc()
        print("\tDone.", flush=True)
        # get params
        print("Running acpype:", end="", flush=True)
        if not self._check_done(self.PCC_dir/"PCC.acpype"):
            self._get_params()
        print("\tDone.", flush=True)
        # minimize PCC
        print("Minimizing PCC:", end="", flush=True)
        if not self._check_done(self.PCC_dir/'em'):
            self._minimize_PCC()
        print("\tDone.", flush=True)
        # create and minimize the complex
        print("Creating complex box and minimizing:", end="", flush=True)
        self._mix()
        print("\tDone.", flush=True)
        # run PBMetaD
        print("Running PBMetaD:", end="", flush=True)
        if not self._check_done(self.complex_dir/'pbmetad'):
            self._pbmetaD()
        print("\tDone.", flush=True)
        # reweight
        print("Reweighting PBMetaD results:", end="", flush=True)
        if not self._check_done(self.complex_dir/'reweight'):
            self._reweight()
        print("\tDone.", flush=True)
        # calc FE
        self.free_e, self.free_e_err = self._calc_FE()
        return self.free_e, self.free_e_err
