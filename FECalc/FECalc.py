import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

from .GMXitp.GMXitp import GMXitp
from .utils import cd

class FECalc():
    """
    Class to calculate free energy surface given PCC peptide chain and target name.
    First, the PCC is generated from the master PCC structure. Then `AMBER` parameters
    are generated for the new PCC using `acpype`. New PCC is then solvated and equilibrated.
    The equilibrated structures is placed in a box with the target molecule using `packmol`.
    The complex box is then solvated and equilibrated and the free energy surface is calculated
    using `PBMetaD`.
    """
    def __init__(self, pcc, target, base_dir: Path, temp: float, box: float, **kwargs) -> None:
        """
        Setup the base, PCC, and complex directories, and locate the target molecule files.
    
        Args:
            pcc (PCCBuilder): PCC structure for FE calculations.
            target (TargetMol): Target molecule for FE calculations.
            base_dir (Path): directory to store the calculations
            temp (float): Temperature of the simulations
            box (float): Size of the simulation box

        Raises:
            ValueError: Raises Value error if `base_dir` is not a directory.
        """
        self.pcc = pcc
        self.target = target
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Free energy calculations for {self.pcc.PCC_code} with {self.target.name} (PID: {os.getpid()})")
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

        self.PCC_dir = self.pcc.PCC_dir # directory to store PCC calculations

        self.complex_dir = self.base_dir/"complex" # directory to store complex calculations
        self.complex_dir.mkdir(exist_ok=True)

        self.target_dir = self.target.base_dir/"export"

        self.PCC_charge = self.pcc.charge
        self.MOL_list = [] # list of MOL atom ids (str)
        self.PCC_list = [] # list of PCC atom ids (str)
        self.MOL_list_atom = [] # list of MOL atom names (str)
        self.PCC_list_atom = [] # list of PCC atom names (str)
        self.T = float(temp)
        self.KbT = 8.314 * self.T
        self.box_size = float(box)

        # MetaD setup
        self.n_steps = int(kwargs.get("n_steps", 400000000))
        self.metad_height = float(kwargs.get("metad_height", 3.0))
        self.metad_pace = int(kwargs.get("metad_pace", 500))
        self.metad_bias_factor = float(kwargs.get("metad_bias_factor", 20))
    
    def _check_done(self, stage: Path) -> bool:
        """
        Check if a calculation stage has already been performed.

        Args:
            stage (Path): Directory for the calculation stage.

        Returns:
            bool: True if the stage has a ".done" file, False otherwise.
        """
        stage_path = Path(stage)
        if not stage_path.is_absolute():
            stage_path = self.base_dir / stage_path
        done_file = stage_path / ".done"
        return done_file.exists()

    def _set_done(self, stage: Path) -> None:
        """
        Create an empty ".done" file in the stage directory to mark it as completed.

        Args:
            stage (Path): Directory for the completed stage.

        Returns:
            None
        """
        stage_path = Path(stage)
        if not stage_path.is_absolute():
            stage_path = self.base_dir / stage_path
        stage_path.mkdir(parents=True, exist_ok=True)
        done_file = stage_path / ".done"
        done_file.touch()
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
        """Regenerate position restraints using updated atom indices.

        Reads atom indices from the minimized ``em.gro`` structure and rewrites
        ``posre_MOL.itp`` and ``posre_PCC.itp`` so that restraints apply to the
        correct atoms after mixing.

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
    
    def update_mdp(self, mdp_in, mdp_out, n_steps=None):
        """
        Update the mdp files with the correct temperature and optionally the number of steps.

        Args:
            mdp_in (Path): Path to the tempelate
            mdp_out (Path): Path to the output file
            n_steps (int, optional): Number of steps to set in the output. If ``None`` the
                steps value in the template is kept unchanged.
        """
        lines = []
        with open(mdp_in, 'r') as f:
            for line in f:
                if line.strip().startswith('ref_t'):
                    line = f'ref_t              = {self.T}     {self.T}\n'
                elif line.strip().startswith('gen_temp'):
                    line = f'gen_temp           = {self.T}\n'
                elif line.strip().startswith('nsteps') and n_steps is not None:
                    line = f'nsteps           = {n_steps}\n'
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
                subprocess.run(["cp", f"{self.target_dir}/MOL.itp", "."], check=True)
                subprocess.run(["cp", f"{self.target_dir}/MOL.pdb", "."], check=True)
                subprocess.run(["cp", f"{self.target_dir}/posre_MOL.itp", "."], check=True) # This has incorrect atom numbers
                subprocess.run(["cp", f"{self.PCC_dir}/PCC.acpype/PCC_GMX.itp", "./PCC.itp"], check=True)
                subprocess.run(["cp", f"{self.PCC_dir}/PCC.acpype/posre_PCC.itp", "."], check=True)
                subprocess.run(["cp", f"{self.PCC_dir}/em/PCC_em.pdb", "./PCC.pdb"], check=True)
                # create complex.pdb with packmol
                subprocess.run(["cp", f"{self.mold_dir}/complex/mix/mix.inp", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/mix/run_packmol.sh", "."], check=True)
                subprocess.run("bash -c 'source run_packmol.sh'", shell=True, check=True)
                # check for complex.pdb
                if not Path.exists(self.complex_dir/"complex.pdb"):
                    raise RuntimeError(f"Packmol output not found. Check {self.complex_dir}.")
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
                subprocess.run(["cp", "../MOL_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_MOL.itp", "."], check=True)
                subprocess.run(["cp", "../PCC_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_PCC.itp", "."], check=True)
                subprocess.run(["cp", "../complex.itp", "."], check=True)
                subprocess.run(["cp", "../complex.pdb", "."], check=True)
                subprocess.run(["cp", "../topol.top", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/em/ions.mdp", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/em/em.mdp", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/em/sub_mdrun_complex_em.sh", "."], check=True) # copy mdrun submission script
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.pcc.PCC_code}{wait_str}sub_mdrun_complex_em.sh {self.box_size} {self.pcc.charge}", check=True, shell=True)
            self._set_done(self.complex_dir/'em')

        if not self.MOL_list:
            with cd(self.complex_dir/"em"): # cd into complex/em
                # update atom ids
                self._get_atom_ids("./em.gro")
            # regenerate posre files with updated atom ids
            with cd(self.complex_dir):
                self._fix_posre()
        ## NVT
        if not self._check_done(self.complex_dir/"nvt"):
            # create complex/nvt dir
            Path.mkdir(self.complex_dir/"nvt", exist_ok=True)
            with cd(self.complex_dir/"nvt"): # cd into complex/nvt
                # copy files into complex/nvt
                subprocess.run(["cp", "../MOL_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../PCC_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../complex.itp", "."], check=True)
                subprocess.run(["cp", "../posre_MOL.itp", "."], check=True)
                subprocess.run(["cp", "../posre_PCC.itp", "."], check=True)
                subprocess.run(["cp", "../em/topol.top", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/nvt/sub_mdrun_complex_nvt.sh", "."], check=True) # copy mdrun submission script
                # copy nvt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/nvt/nvt.mdp", "./nvt_temp.mdp"], check=True)
                else:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/nvt/nvt_nions.mdp", "./nvt_temp.mdp"], check=True)
                # set temperature
                self.update_mdp("./nvt_temp.mdp", "./nvt.mdp")
                subprocess.run(f"rm ./nvt_temp.mdp", shell=True)
                # submit nvt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.pcc.PCC_code}{wait_str}sub_mdrun_complex_nvt.sh", check=True, shell=True)
            self._set_done(self.complex_dir/'nvt')
        ## NPT
        if not self._check_done(self.complex_dir/"npt"):
            # create complex/npt dir
            Path.mkdir(self.complex_dir/"npt", exist_ok=True)
            with cd(self.complex_dir/"npt"): # cd into complex/npt
                # copy files into complex/npt
                subprocess.run(["cp", "../MOL_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_MOL.itp", "."], check=True)
                subprocess.run(["cp", "../PCC_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_PCC.itp", "."], check=True)
                subprocess.run(["cp", "../complex.itp", "."], check=True)
                subprocess.run(["cp", "../nvt/topol.top", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/npt/sub_mdrun_complex_npt.sh", "."], check=True) # copy mdrun submission script
                # copy npt.mdp into nvt
                if self.PCC_charge != 0:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/npt/npt.mdp", "./npt_temp.mdp"], check=True)
                else:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/npt/npt_nions.mdp", "./npt_temp.mdp"], check=True)
                # set temperature
                self.update_mdp("./npt_temp.mdp", "./npt.mdp")
                subprocess.run(f"rm ./npt_temp.mdp", shell=True)
                # submit npt job
                wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
                subprocess.run(f"sbatch -J {self.pcc.PCC_code}{wait_str}sub_mdrun_complex_npt.sh", shell=True)
            self._set_done(self.complex_dir/'npt')
        return

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
        replacements = {
            "${11}": self.metad_bias_factor,
            "${10}": self.metad_pace,
            "${9}": self.T,
            "${8}": self.metad_height,
            "${7}": vrb_atom_ids,
            "${6}": v2b_atom_ids,
            "${5}": v2a_atom_ids,
            "${4}": v1b_atom_ids,
            "${3}": v1a_atom_ids,
            "${2}": MOL_atom_id,
            "${1}": PCC_atom_id,
        }

        for i, line in enumerate(cnt):
            for key, value in replacements.items():
                line = line.replace(key, str(value))
            cnt[i] = line
        
        # write new plumed file
        with open(plumed_out, 'w') as f:
            f.writelines(cnt)

        return None

    def _pbmetaD(self, wait: bool = True) -> None:
        """Run Parallel Bias Metadynamics on the equilibrated complex.

        PBMetaD enhances sampling by depositing Gaussian bias potentials along
        predefined collective variables (CVs). The resulting history-dependent
        bias reconstructs the underlying free-energy surface. The simulation can
        be resumed from checkpoints and optionally blocks until completion.

        Args:
            wait (bool, optional): Whether to wait for the job to finish.
                Defaults to ``True``.

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
                subprocess.run(["cp", "./md.cpt", "./md.cpt.bck.unk"], check=True)
                # check that all GRID files exist. If not replace them with backups.
                if not Path.exists(self.complex_dir/"md"/"GRID_COM"):
                    print(f"{now}: Missing GRID_COM file. Replacing with latest backup.", flush=True)
                    subprocess.run(["cp", "./bck.last.GRID_COM", "./GRID_COM"], check=True)
                if not Path.exists(self.complex_dir/"md"/"GRID_cos"):
                    print(f"{now}: Missing GRID_cos file. Replacing with latest backup.", flush=True)
                    subprocess.run(["cp", "./bck.last.GRID_cos", "./GRID_cos"], check=True)
                if not Path.exists(self.complex_dir/"md"/"GRID_ang"):
                    print(f"{now}: Missing GRID_ang file. Replacing with latest backup.", flush=True)
                    subprocess.run(["cp", "./bck.last.GRID_ang", "./GRID_ang"], check=True)
            else:
                # copy files into complex/pbmetad
                subprocess.run(["cp", "../MOL_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_MOL.itp", "."], check=True)
                subprocess.run(["cp", "../PCC_truncated.itp", "."], check=True)
                subprocess.run(["cp", "../posre_PCC.itp", "."], check=True)
                subprocess.run(["cp", "../complex.itp", "."], check=True)
                subprocess.run(["cp", "../npt/topol.top", "."], check=True)
                subprocess.run(["cp", f"{self.mold_dir}/complex/md/sub_mdrun_plumed.sh", "."], check=True) # copy mdrun submission script
                subprocess.run(["cp", f"{self.mold_dir}/complex/md/plumed.dat", "./plumed_temp.dat"], check=True) # copy pbmetad script
                subprocess.run(["cp", f"{self.mold_dir}/complex/md/plumed_restart.dat", "./plumed_r_temp.dat"], check=True) # copy pbmetad script
                # update PCC and MOL atom ids
                self._create_plumed("./plumed_temp.dat", "./plumed.dat")
                self._create_plumed("./plumed_r_temp.dat", "./plumed_restart.dat")
                # remove temp plumed file
                subprocess.run(f"rm ./plumed_temp.dat", shell=True)
                subprocess.run(f"rm ./plumed_r_temp.dat", shell=True)
                # copy nvt.mdp into pbmetad
                if self.PCC_charge != 0:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/md/md.mdp", "./md_temp.mdp"], check=True)
                else:
                    subprocess.run(["cp", f"{self.mold_dir}/complex/md/md_nions.mdp", "./md_temp.mdp"], check=True)
                # set temperature and, if requested, the number of steps
                self.update_mdp("./md_temp.mdp", "./md.mdp", n_steps=self.n_steps)
                subprocess.run(f"rm ./md_temp.mdp", shell=True)
                
            # submit pbmetad job with limited retries
            max_attempts = 5
            attempt = 0
            while attempt < max_attempts:
                if attempt > 0:
                    subprocess.run(f"mv ./HILLS_ang ./HILLS_ang.bck.{attempt}", shell=True, check=False)
                    subprocess.run(f"mv ./HILLS_cos ./HILLS_cos.bck.{attempt}", shell=True, check=False)
                    subprocess.run(f"mv ./HILLS_COM ./HILLS_COM.bck.{attempt}", shell=True, check=False)
                    subprocess.run(["cp", f"./GRID_ang", f"./GRID_ang.bck.{attempt}"], check=False)
                    subprocess.run(["cp", f"./GRID_cos", f"./GRID_cos.bck.{attempt}"], check=False)
                    subprocess.run(["cp", f"./GRID_COM", f"./GRID_COM.bck.{attempt}"], check=False)
                    subprocess.run(f"cp ./md.cpt ./md.cpt.bck.{attempt}", shell=True, check=False)
                    now = datetime.now()
                    now = now.strftime("%m/%d/%Y, %H:%M:%S")
                    print(f"{now}: Resubmitting PBMetaD: ", end="", flush=True)
                try:
                    subprocess.run(
                        f"sbatch -J {self.pcc.PCC_code}{wait_str}sub_mdrun_plumed.sh",
                        check=True,
                        shell=True,
                    )
                    if attempt > 0:
                        print()
                    break
                except subprocess.CalledProcessError as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"PBMetaD run failed after {max_attempts} attempts: {e}"
                        ) from e

            # ensure workflow completed successfully
            if not Path.exists(self.complex_dir/"md"/"md.gro"):
                raise RuntimeError("The PBMetaD run did not complete successfully. Check the PBMetaD directory for more information and rerun FECalc to continue or retry.")
    
        self._set_done(self.complex_dir/'md')
        return None
    
    def _reweight(self, wait: bool = True) -> None:
        """Reweight biased trajectories to recover unbiased observables.

        The method applies the final metadynamics bias to generate a
        probability distribution for the collective variables free from the
        influence of the adaptive bias potential. This is achieved by using a
        ``plumed`` reweighting script that recalculates the final, converged
        biased using the grid files from the PBMetaD simulation and creates a
        new COLVARS file with the converged values of bias which is used by
        ``postprocess`` to calculate the free energy.

        Args:
            wait (bool, optional): Whether to wait for the reweighting job to
                complete. Defaults to ``True``.

        Returns:
            None
        """
        # create complex/reweight dir
        Path.mkdir(self.complex_dir/"reweight", exist_ok=True)
        with cd(self.complex_dir/"reweight"): # cd into complex/reweight
            # copy files into complex/reweight
            #subprocess.run("cp ../pbmetad/HILLS_COM .", shell=True, check=True)
            #subprocess.run("cp ../pbmetad/HILLS_ang .", shell=True, check=True)
            subprocess.run(["cp", "../md/GRID_COM", "."], check=True)
            subprocess.run(["cp", "../md/GRID_ang", "."], check=True)
            subprocess.run(["cp", "../md/GRID_cos", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/complex/reweight/sub_mdrun_rerun.sh", "."], check=True) # copy mdrun submission script
            subprocess.run(["cp", f"{self.mold_dir}/complex/reweight/reweight.dat", "./reweight_temp.dat"], check=True) # copy reweight script
            # update PCC and MOL atom ids
            self._create_plumed("./reweight_temp.dat", "./reweight.dat")
            # remove temp plumed file
            subprocess.run(f"rm ./reweight_temp.dat", shell=True)
            # submit reweight job
            wait_str = " --wait " if wait else "" # whether to wait for reweight to finish before exiting
            subprocess.run(f"sbatch -J {self.pcc.PCC_code}{wait_str}sub_mdrun_rerun.sh", check=True, shell=True)
        self._set_done(self.complex_dir/'reweight')
        return None

    def run(self, n_steps=None) -> tuple:
        """Wrapper for FE calculations. Create PCC, call acpype,
        minimize, create complex, and run PBMetaD.

        Args:
            n_steps (int, optional): Override the number of PBMetaD simulation steps.
        """
        if n_steps is not None:
            self.n_steps = n_steps
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
