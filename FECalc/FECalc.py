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
    def __init__(self, pcc, target, base_dir: Path, temp: float) -> None:
        """
        Setup the base, PCC, and complex directories, and locate the target molecule files.
    
        Args:
            pcc (PCCBuilder): PCC structure for FE calculations.
            target (TargetMol): Target molecule for FE calculations.
            base_dir (Path): directory to store the calculations
            script_dir (Path): directory containing necessary scripts and templates

        Raises:
            ValueError: Raises Value error if `base_dir` is not a directory.
        """
        self.pcc = pcc
        self.target = target
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Free energy calculations for {self.PCC_code} with {self.target} (PID: {os.getpid()})")
        self.script_dir = Path(__file__).parent/Path("scripts")#Path(self.settings['scripts_dir'])
        self.mold_dir = Path(__file__).parent/Path("mold")
        self.base_dir = Path(base_dir) # base directory to store files
        if self.base_dir.exists():
            if not self.base_dir.is_dir():
                raise ValueError(f"{self.base_dir} is not a directory.")
        else:
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Base directory does not exist. Creating...")
            self.base_dir.mkdir()

        self.PCC_dir = self.pcc.base_dir # directory to store PCC calculations

        self.complex_dir = self.base_dir/"complex" # directory to store complex calculations
        self.complex_dir.mkdir(exist_ok=True)

        self.target_dir = self.target.base_dir/"export"

        self.PCC_charge = self.pcc.charge
        self.PCC_n_atoms = self.pcc.n_atoms
        self.MOL_list = [] # list of MOL atom ids (str)
        self.PCC_list = [] # list of PCC atom ids (str)
        self.MOL_list_atom = [] # list of MOL atom names (str)
        self.PCC_list_atom = [] # list of PCC atom names (str)
        self.T = float(temp)
        self.KbT = self.T * 8.314 # System temperature in J/mol
    
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
    
    def update_temperature(self, mdp_in, mdp_out):
        lines = []
        with open(mdp_in, 'r') as f:
            for line in f:
                if line.strip().startswith('ref_t'):
                    line = f'ref_t              = {self.T}     {self.T}\n'
                elif line.strip().startswith('gen_temp'):
                    line = f'gen_temp           = {self.T}\n'
                lines.append(line)

        with open(mdp_out, 'w') as f:
            f.writelines(lines)

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
                subprocess.run(f"cp {self.mold_dir}/complex/mix/run_packmol.sh .", shell=True, check=True)
                subprocess.run("bash -c 'source run_packmol.sh'", shell=True, check=True)
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
                    subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt.mdp ./nvt_temp.mdp", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt_nions.mdp ./nvt_temp.mdp", shell=True, check=True)
                # set temperature
                self.update_temperature("./nvt_temp.mdp", "./nvt.mdp")
                subprocess.run(f"rm ./nvt_temp.mdp", shell=True)
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
                    subprocess.run(f"cp {self.mold_dir}/complex/npt/npt.mdp ./npt_temp.mdp", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/npt/npt_nions.mdp ./npt_temp.mdp", shell=True, check=True)
                # set temperature
                self.update_temperature("./npt_temp.mdp", "./npt.mdp")
                subprocess.run(f"rm ./npt_temp.mdp", shell=True)
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
        a_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in self.pcc.origin]
        b_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in self.pcc.anchor_point1]
        v1a_atom_ids = "".join([f"{i}," for i in a_list])[:-1]
        v1b_atom_ids = "".join([f"{i}," for i in b_list])[:-1]
        b_list = [self.PCC_list[self.PCC_list_atom.index(i)] for i in self.pcc.anchor_point2]
        vrb_atom_ids = "".join([f"{i}," for i in b_list])[:-1]
        a_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in self.target.anchor_point1]
        b_list = [self.MOL_list[self.MOL_list_atom.index(i)] for i in self.target.anchor_point2]
        
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
            if Path.exists(self.complex_dir/"md"/"md.cpt"): # if there's a checkpoint, continue the run
                now = datetime.now()
                now = now.strftime("%m/%d/%Y, %H:%M:%S")
                print(f"{now}: Resuming previous run...", flush=True)
                subprocess.run("mv ./HILLS_ang ./HILLS_ang.bck.unk", shell=True, check=False)
                subprocess.run("mv ./HILLS_cos ./HILLS_cos.bck.unk", shell=True, check=False)
                subprocess.run("mv ./HILLS_COM ./HILLS_COM.bck.unk", shell=True, check=False)
                subprocess.run("cp ./md.cpt ./md.cpt.bck.unk", shell=True, check=True)
                # check that all GRID files exist. If not replace them with backups.
                if not Path.exists(self.complex_dir/"md"/"GRID_COM"):
                    print(f"{now}: Missing GRID_COM file. Replacing with latest backup.", flush=True)
                    subprocess.run("cp ./bck.last.GRID_COM ./GRID_COM", shell=True, check=True)
                if not Path.exists(self.complex_dir/"md"/"GRID_cos"):
                    print(f"{now}: Missing GRID_cos file. Replacing with latest backup.", flush=True)
                    subprocess.run("cp ./bck.last.GRID_cos ./GRID_cos", shell=True, check=True)
                if not Path.exists(self.complex_dir/"md"/"GRID_ang"):
                    print(f"{now}: Missing GRID_ang file. Replacing with latest backup.", flush=True)
                    subprocess.run("cp ./bck.last.GRID_ang ./GRID_ang", shell=True, check=True)
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
                    subprocess.run(f"cp {self.mold_dir}/complex/md/md.mdp ./md_temp.mdp", shell=True, check=True)
                else:
                    subprocess.run(f"cp {self.mold_dir}/complex/md/md_nions.mdp ./md_temp.mdp", shell=True, check=True)
                # set temperature
                self.update_temperature("./md_temp.mdp", "./md.mdp")
                subprocess.run(f"rm ./md_temp.mdp", shell=True)
                
            # submit pbmetad job. Resubmits until either the job fails 10 times or it succesfully finishes.
            cnt = 1
            try:
                subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
                if not Path.exists(self.complex_dir/"md"/"md.gro"): # making sure except block is executed if the run is not complete, regardless of system exit code
                    raise RuntimeError("Run not completed.")
            except:
                fail_flag = True
                while fail_flag:
                    try:
                        cnt += 1
                        subprocess.run(f"mv ./HILLS_ang ./HILLS_ang.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"mv ./HILLS_cos ./HILLS_cos.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"mv ./HILLS_COM ./HILLS_COM.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"cp ./GRID_ang ./GRID_ang.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"cp ./GRID_cos ./GRID_cos.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"cp ./GRID_COM ./GRID_COM.bck.{cnt}", shell=True, check=False)
                        subprocess.run(f"cp ./md.cpt ./md.cpt.bck.{cnt}", shell=True, check=False)
                        now = datetime.now()
                        now = now.strftime("%m/%d/%Y, %H:%M:%S")
                        print(f"{now}: Resubmitting PBMetaD: ", end="", flush=True)
                        subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
                        print()
                        fail_flag = False
                    except:
                        if cnt >= 5:
                            raise RuntimeError("PBMetaD run failed more than 5 times. Stopping.")
    
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

    def run(self) -> tuple:
        """Wrapper for FE caclculations. Create PCC, call acpype, 
        minimize, create complex, and run PBMetaD.
        """
        # create PCC
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Running pymol: ", end="", flush=True)
        if not self._check_done(self.PCC_dir):
            self._create_pcc()
            print("Check the initial structure and rerun the code.")
            return self.free_e, self.free_e_err, self.K, self.K_err
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
        print(f"{now}: All steps completed.")
        print("-"*30 + "Finished" + "-"*30)
        
        return None
