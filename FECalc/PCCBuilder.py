import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

from .utils import cd, _read_pdb, _write_coords_to_pdb, _prep_pdb


class PCCBuilder():
    """Construct and minimize peptide-based core conjugates (PCCs).

    The resulting peptides are composed of D-amino acids and serve as the
    scaffold for the PCC. The class mutates a reference PCC structure to match a
    desired amino-acid sequence, generates force-field parameters using
    ``acpype`` and produces an equilibrated peptide ready to be combined with a
    target molecule.
    """

    def __init__(self, pcc: str, base_dir: Path, settings_json: Path) -> None:
        """Initialize builder and create working directories.

        Args:
            pcc (str): Amino-acid sequence (single-letter code) from arm to bead.
            base_dir (Path): Directory in which calculation files will be stored.
            settings_json (Path): Path to a JSON file with build settings.

        Raises:
            ValueError: If ``base_dir`` exists and is not a directory.
        """
        self.PCC_code = pcc
        now = datetime.now()
        now = now.strftime("%m/%d/%Y, %H:%M:%S")
        print(f"{now}: Building and minimizing structure for {self.PCC_code} (PID: {os.getpid()})")
        self.settings_dir = Path(settings_json) # path settings.JSON
        with open(self.settings_dir) as f:
            self.settings = json.load(f)
        self.script_dir = Path(__file__).parent/"scripts"
        self.mold_dir = Path(__file__).parent/"mold"
        self.base_dir = Path(base_dir) # base directory to store files
        if self.base_dir.exists():
            if not self.base_dir.is_dir():
                raise ValueError(f"{self.base_dir} is not a directory.")
        else:
            now = datetime.now()
            now = now.strftime("%m/%d/%Y, %H:%M:%S")
            print(f"{now}: Base directory does not exist. Creating...")
            self.base_dir.mkdir()

        self.PCC_dir = self.base_dir/f"{self.PCC_code}" # directory to store PCC calculations
        self.PCC_dir.mkdir(exist_ok=True)

        self.PCC_ref = Path(self.settings["ref_PCC_dir"]) # path to refrence PCC structure
        
        self.AAdict31 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'} # 3 to 1 translator
        self.AAdict13 = {j: i for i, j in self.AAdict31.items()} # 1 to 3 translator
        self.charge_dict = {"D": -1, "E": -1, "R": +1, "K": +1} # AA charges at neutral pH

        self.charge = sum([self.charge_dict.get(i, 0) for i in list(self.PCC_code)]) # Net charge of the PCC at pH=7
        self.n_atoms = None # number of PCC atoms
        self.origin = self.settings["origin"]
        self.anchor_point1 = self.settings["anchor1"]
        self.anchor_point2 = self.settings["anchor2"]

        self.pymol = Path(self.settings['pymol_dir']) # path to pymol installation

    def _check_done(self, stage: Path) -> bool:
        """
        Check if a calculation stage has been performed already.

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

    def _create_pcc(self) -> None:
        """Generate the mutated PCC structure using PyMOL.

        The reference PCC is mutated residue-by-residue through the auxiliary
        ``PCCmold.py`` script. Residues are converted to their D-amino acid
        counterparts, and a short pre-optimization step removes clashes
        introduced by the mutations.

        Returns:
            None
        """
        pcc_translate = [self.AAdict13[i] for i in list(self.PCC_code)] # translate the 1 letter AA code to 3 letter code
        numres = len(pcc_translate) # number of residues
        mutation = "".join(pcc_translate) # concatenate the 3 letter codes
        subprocess.run(f"{self.pymol} -qc {self.script_dir}/PCCmold.py -i {self.PCC_ref} -o {self.PCC_dir/self.PCC_code}.pdb -r {numres} -m {mutation}", shell=True, check=True)
        _, _, coords = _read_pdb(self.PCC_dir/f"{self.PCC_code}.pdb")
        self.n_atoms = coords.shape[0]
        # pre-optimization
        wait_str = " --wait "
        subprocess.run(["cp", f"{self.mold_dir}/PCC/sub_preopt.sh", f"{self.PCC_dir}"], check=True) # copy preopt submission script
        with cd(self.PCC_dir): # cd into the PCC directory
            # pre-optimize to deal with possible clashes created while changing residues to D-AAs
            print("Pre-optimizing: ", flush=True)
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_preopt.sh {self.PCC_code}.pdb {self.PCC_code}_babel.pdb", shell=True, check=True)
            _, _, coords_new = _read_pdb(f"{self.PCC_code}_babel.pdb")
            _write_coords_to_pdb(f"{self.PCC_code}.pdb", f"{self.PCC_code}_opt.pdb", coords_new[:self.n_atoms, ...])

        self._set_done(self.PCC_dir) # mark stage as done
        return None
    
    def _get_params(self, wait: bool = True) -> None:
        """Generate GAFF parameters for the PCC using ``acpype``.

        The method performs the following steps:

        1. Copy the ``sub_acpype.sh`` submission script into the PCC working
           directory.
        2. Convert the optimized PCC structure into a single-residue PDB so
           that ``acpype`` can assign parameters.
        3. Launch ``acpype`` through the submission script. When ``wait`` is
           ``True`` the call blocks until the job finishes.
        4. Inspect ``PCC.acpype/acpype.log`` for any warnings. If the log
           contains the word ``warning`` (case insensitive) a ``RuntimeError``
           is raised because ``acpype`` likely produced an inconsistent bonded
           topology.

        Args:
            wait (bool, optional): Whether to block until ``acpype`` finishes.
                Defaults to ``True``.

        Raises:
            RuntimeError: If the ``acpype`` log contains warnings that may
                indicate incorrect topology generation.

        Returns:
            None
        """
        
        subprocess.run(["cp", f"{self.mold_dir}/PCC/sub_acpype.sh", f"{self.PCC_dir}"], check=True) # copy acpype submission script

        wait_str = " --wait " if wait else "" # whether to wait for acpype to finish before exiting        
        with cd(self.PCC_dir): # cd into the PCC directory
                        # create acpype pdb with 1 residue
            _prep_pdb(f"{self.PCC_code}_opt.pdb", f"{self.PCC_code}_acpype.pdb", "PCC")
            # run acpype
            print("Running acpype: ", flush=True)
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_acpype.sh {self.PCC_code}_acpype PCC {self.charge}", shell=True, check=True)
        
            # check acpype.log for warnings
            with open("PCC.acpype/acpype.log") as f:
                acpype_log = f.read()
                if "warning:" in acpype_log.lower():
                    raise RuntimeError("""Acpype generated files are likely to have incorrect bonds. 
                                       Check the generated PCC structure before continueing.""")

        self._set_done(self.PCC_dir/"PCC.acpype")
        return None
    
    def _minimize_PCC(self, wait: bool = True) -> None:
        """Solvate and minimize the PCC structure.

        Acpype-generated topology files are combined with a solvent box and the
        system is minimized using GROMACS. The equilibrated structure
        ``PCC.gro`` is saved for subsequent complex assembly.

        Args:
            wait (bool, optional): Whether to wait for the minimization to
                finish. Defaults to ``True``.

        Returns:
            None
        """
        Path.mkdir(self.PCC_dir/"em", exist_ok=True)
        with cd(self.PCC_dir/"em"): # cd into PCC/em
            # copy acpype files into em dir
            subprocess.run(["cp", "../PCC.acpype/PCC_GMX.gro", "."], check=True)
            subprocess.run(["cp", "../PCC.acpype/PCC_GMX.itp", "."], check=True)
            subprocess.run(["cp", "../PCC.acpype/posre_PCC.itp", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/topol.top", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/ions.mdp", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/em.mdp", "."], check=True)
            subprocess.run(["cp", f"{self.mold_dir}/PCC/em/sub_mdrun_em.sh", "."], check=True) # copy mdrun submission script
            # submit em job
            wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_em.sh PCC {self.charge}", check=True, shell=True)
        self._set_done(self.PCC_dir/"em")
        return None

    def create(self) -> tuple:
        """Run the full peptide construction workflow.

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
