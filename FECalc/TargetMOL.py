import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

from .utils import cd, _prep_pdb


class TargetMOL():
    """Handle preparation of the small-molecule target.

    The class loads user-provided settings, generates force-field parameters
    with ``acpype``, minimizes the structure, and exports the files required
    for complex assembly with a PCC.
    """

    def __init__(self, settings_json: Path) -> None:
        """Initialize the target molecule from a settings file.

        Args:
            settings_json (Path): Path to a JSON file with target
                configuration and I/O paths.
        """

        with open(Path(settings_json)) as f:
            self.settings = json.load(f)

        self.name = self.settings["name"]
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Building and minimizing structure for {self.name} (PID: {os.getpid()})")
        
        self.mold_dir = Path(__file__).parent / Path("mold")

        output_dir = self.settings.get("output_dir")
        if not output_dir:
            raise ValueError("'output_dir' must be specified and cannot be empty.")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(
                f"Output path '{self.output_dir}' exists and is not a directory."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = self.output_dir  # backwards compatibility

        input_pdb_dir = self.settings.get("input_pdb_dir")
        if not input_pdb_dir:
            raise ValueError("'input_pdb_dir' must be specified and cannot be empty.")
        self.input_pdb_dir = Path(input_pdb_dir)
        if not self.input_pdb_dir.exists() or not self.input_pdb_dir.is_file():
            raise ValueError(
                f"Input PDB file '{self.input_pdb_dir}' does not exist or is not a file."
            )

        self.charge = int(self.settings.get("charge", 0))
        self.anchor_point1 = self.settings["anchor1"]
        self.anchor_point2 = self.settings["anchor2"]

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
    
    def _get_params(self) -> None:
        """Generate GAFF parameters for the target molecule using ``acpype``.

        ``acpype`` is executed directly instead of through the previous
        ``sub_acpype.sh`` submission script.

        Raises:
            RuntimeError: If ``acpype.log`` contains warnings indicating
                potential problems with the generated topology.

        Returns:
            None
        """
        # copy the input pdb to working dir
        subprocess.run(
            f"cp {self.input_pdb_dir} {self.base_dir}/MOL.pdb", shell=True, check=True
        )

        with cd(self.base_dir):  # cd into the working directory
            # create acpype pdb with 1 residue
            _prep_pdb("MOL.pdb", "MOL_acpype.pdb", "MOL")

            # run acpype directly
            acpype_cmd = (
                f"acpype -i MOL_acpype.pdb -b MOL -c bcc -n {self.charge} -a gaff2"
            )
            print("Running acpype: ", flush=True)
            subprocess.run(acpype_cmd, shell=True, check=True)

            # check acpype.log for warnings
            with open("MOL.acpype/acpype.log") as f:
                acpype_log = f.read()
                if "warning:" in acpype_log.lower():
                    raise RuntimeError(
                        """Acpype generated files are likely to have incorrect bonds.
                                       Check the generated MOL structure before continueing."""
                    )

        self._set_done(self.base_dir / "MOL.acpype")
        return None
    
    def _minimize_MOL(self, wait: bool = True) -> None: 
        """
        Run minimization for MOL. Copies acpype files into `em` directory, solvates, adds ions, and minimizes
        the structure. The final coordinates are converted from `em.gro` to `MOL_em.pdb`.

        Args:
            wait (bool, optional): Whether to wait for `em` to finish. Defaults to True.

        Returns:
            None
        """
        Path.mkdir(self.base_dir/"em", exist_ok=True)
        with cd(self.base_dir/"em"): # cd into em
            # copy acpype files into em dir
            subprocess.run(["cp", "../MOL.acpype/MOL_GMX.gro", "."], check=True)
            subprocess.run(["cp", "../MOL.acpype/MOL_GMX.itp", "."], check=True)
            subprocess.run(["cp", "../MOL.acpype/posre_MOL.itp", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/topol.top", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/ions.mdp", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/em.mdp", "."], check=True)
            # fix topol.top
            subprocess.run(f"sed -i 's/PCC/MOL/g' topol.top", shell=True)

            # Determine total number of threads for mdrun from Slurm env vars
            ncpu = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
            nthr = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            nnod = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
            np = ncpu * nthr * nnod

            # Create box
            subprocess.run([
                "gmx", "editconf", "-f", "MOL_GMX.gro", "-o", "MOL_box.gro", "-c", "-d", "1.0", "-bt", "cubic"
            ], check=True)

            # Solvate
            subprocess.run([
                "gmx", "solvate", "-cp", "MOL_box.gro", "-cs", "spc216.gro", "-o", "MOL_sol.gro", "-p", "topol.top"
            ], check=True)

            # Neutralize if needed and prepare tpr
            if self.charge != 0:
                subprocess.run([
                    "gmx", "grompp", "-f", "ions.mdp", "-c", "MOL_sol.gro", "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "2"
                ], check=True)
                subprocess.run([
                    "gmx", "genion", "-s", "ions.tpr", "-o", "MOL_sol_ions.gro", "-p", "topol.top", "-pname", "NA", "-nname", "CL", "-neutral"
                ], input="4\n", text=True, check=True)
                subprocess.run([
                    "gmx", "grompp", "-f", "em.mdp", "-c", "MOL_sol_ions.gro", "-p", "topol.top", "-o", "em.tpr"
                ], check=True)
            else:
                subprocess.run([
                    "gmx", "grompp", "-f", "em.mdp", "-c", "MOL_sol.gro", "-p", "topol.top", "-o", "em.tpr"
                ], check=True)

            # Run minimization
            subprocess.run(["gmx", "mdrun", "-ntomp", str(np), "-deffnm", "em"], check=True)

            # Convert minimized structure to PDB
            subprocess.run([
                "gmx", "trjconv", "-s", "em.tpr", "-f", "em.gro", "-o", "MOL_em.pdb", "-pbc", "whole", "-conect"
            ], input="2\n", text=True, check=True)
        self._set_done(self.base_dir/"em")

        return None
    
    def _export(self):
        """Export minimized target files for use in complex assembly.

        Copies the final ``.itp`` and ``.pdb`` files to an ``export``
        directory that can be consumed by :class:`FECalc`.
        """
        Path.mkdir(self.base_dir/"export", exist_ok=True)
        with cd(self.base_dir/"export"):  # cd into export
            subprocess.run(["cp", "../em/MOL_GMX.itp", "./MOL.itp"], check=True)
            subprocess.run(["cp", "../em/posre_MOL.itp", "."], check=True)
            subprocess.run(["cp", "../em/MOL_em.pdb", "./MOL.pdb"], check=True)
        self._set_done(self.base_dir/"export")

    def create(self) -> None:
        """Prepare the target molecule for use in FE calculations.

        This wrapper runs ``acpype`` to generate parameters, minimizes the
        structure, and exports the relevant files. Each step is skipped if a
        ``.done`` marker exists.

        Returns:
            None
        """
        # Check this first in case the simulations were run elsewhere and we
        # are only pointing to the results.
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
            self._export()
            print("\tDone.", flush=True)
            # Done
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: All steps completed.")
        else:
            print(f"{now}: Target molecule loaded from previous calculations.")
        print("-" * 30 + "Finished" + "-" * 30)
        return None

