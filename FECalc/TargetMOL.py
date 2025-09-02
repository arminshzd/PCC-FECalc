import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

from .utils import cd, _prep_pdb


class TargetMOL():
    """_summary_
    """
    def __init__(self, settings_json: Path) -> None:
        """_summary_

        Args:
            target_name (str): name of the target molecule
            base_dir (Path): _description_
            settings_json (Path): _description_
        """

        with open(Path(settings_json)) as f:
            self.settings = json.load(f)

        self.name = self.settings["name"]
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Building and minimizing structure for {self.name} (PID: {os.getpid()})")
        
        self.script_dir = Path(__file__).parent/Path("scripts")
        self.mold_dir = Path(__file__).parent/Path("mold")

        self.base_dir = Path(self.settings['output_dir']) # base directory to store files
        if self.base_dir.exists():
            if not self.base_dir.is_dir():
                raise ValueError(f"{self.base_dir} is not a directory.")
        else:
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Base directory does not exist. Creating...")
            self.base_dir.mkdir()

        self.charge = int(self.settings.get("charge", 0))
        self.anchor_point1 = self.settings["anchor1"]
        self.anchor_point2 = self.settings["anchor2"]
        self.input_pdb_dir = Path(self.settings["input_pdb_dir"]) if \
            self.settings.get("input_pdb_dir", None) is not None else None

    def _check_done(self, stage: Path) -> bool:
        """
        Check if a calculation stage has already been performed.

        Args:
            stage (Path): Directory for the calculation stage.

        Returns:
            bool: True if a ".done" file exists, False otherwise.
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
    
    def _get_n_atoms(self, gro_dir: Path) -> None:
        """
        Get the number of MOL atoms from gro file

        Args:
            gro_dir (Path): path to MOL.gro file.
        """
        with open(gro_dir) as f:
            gro_cnt = f.readlines()
        self.MOL_n_atoms = int(gro_cnt[1].split()[0])
    
    def _get_params(self, wait: bool = True) -> None: 
        """
        Run acpype on the MOL pdb file. Submits a job to the cluster.

        Args:
            wait (bool, optional): Whether to wait for acpype to finish. Defaults to True.

        Returns:
            None
        """
        # copy the input pdb to working dir
        if self.input_pdb_dir is not None:
            subprocess.run(f"cp {self.input_pdb_dir} {self.base_dir}/MOL.pdb", shell=True)
        else:
            raise RuntimeError("Input pdb for calculations was not specified in the settings file.")
        # Copy acpype submission script
        subprocess.run(f"cp {self.mold_dir}/PCC/sub_acpype.sh {self.base_dir}", shell=True)

        # whether to wait for acpype to finish before exiting
        wait_str = " --wait " if wait else ""
        with cd(self.base_dir): # cd into the working directory
                        # create acpype pdb with 1 residue
            _prep_pdb("MOL.pdb", "MOL_acpype.pdb", "MOL")
            # run acpype
            print("Running acpype: ", flush=True)
            subprocess.run(f"sbatch -J MOL{wait_str}sub_acpype.sh MOL_acpype MOL {self.charge}", shell=True, check=True)
        
            # check acpype.log for warnings
            with open("MOL.acpype/acpype.log") as f:
                acpype_log = f.read()
                if "warning:" in acpype_log.lower():
                    raise RuntimeError("""Acpype generated files are likely to have incorrect bonds. 
                                       Check the generated MOL structure before continueing.""")

        self._set_done(self.base_dir/"MOL.acpype")
        return None
    
    def _minimize_MOL(self, wait: bool = True) -> None: 
        """
        Run minimization for MOL. Copies acpype files into `em` directory, solvates, adds ions, and minimizes
        the structure. The last frame is saved as `MOL.gro`

        Args:
            wait (bool, optional): Whether to wait for `em` to finish. Defaults to True.

        Returns:
            None
        """
        Path.mkdir(self.base_dir/"em", exist_ok=True)
        with cd(self.base_dir/"em"): # cd into em
            # copy acpype files into em dir
            subprocess.run("cp ../MOL.acpype/MOL_GMX.gro .", shell=True, check=True)
            subprocess.run("cp ../MOL.acpype/MOL_GMX.itp .", shell=True, check=True)
            subprocess.run("cp ../MOL.acpype/posre_MOL.itp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/topol.top .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/ions.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/em.mdp .", shell=True, check=True)
            subprocess.run(f"cp {self.mold_dir}/PCC/em/sub_mdrun_em.sh .", shell=True) # copy mdrun submission script
            # fix topol.top
            subprocess.run(f"sed -i 's/PCC/MOL/g' topol.top", shell=True)
            # set self.MOL_n_atoms
            #self._get_n_atoms("./PCC_GMX.gro")
            # submit em job
            wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
            subprocess.run(f"sbatch -J MOL{wait_str}sub_mdrun_em.sh MOL {self.charge}", check=True, shell=True)
        self._set_done(self.base_dir/"em")

        return None
    
    def _export(self):
        Path.mkdir(self.base_dir/"export", exist_ok=True)
        with cd(self.base_dir/"export"): # cd into export
            subprocess.run("cp ../em/MOL_GMX.itp ./MOL.itp", shell=True, check=True)
            subprocess.run("cp ../em/posre_MOL.itp .", shell=True, check=True)
            subprocess.run("cp ../em/MOL_em.pdb ./MOL.pdb", shell=True, check=True)
        self._set_done(self.base_dir/"export")

    def create(self) -> None:
        # Check this first incase the simulations were run in a different
        # directory and we are only pointing to the results.
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        if not self._check_done(self.base_dir/'export'):
            print(f"{now}: Getting gaff parameters: ", flush=True)
            if not self._check_done(self.base_dir/"MOL.acpype"):
                self._get_params()
            print("Done.", flush=True)
            # minimize MOL
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Minimizing MOL: ", end="", flush=True)
            if not self._check_done(self.base_dir/'em'):
                self._minimize_MOL()
            print("\tDone.", flush=True)
            # create `export` folder
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Exporting files: ", end="", flush=True)
            if not self._check_done(self.base_dir/'export'):
                self._export()
            print("\tDone.", flush=True)
            # Done
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: All steps completed.")
        else:
            print(f"{now}: Target molecule loaded from previous calculations.")
        print("-"*30 + "Finished" + "-"*30)
        return None