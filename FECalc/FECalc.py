import os
import re
import subprocess
import json
from pathlib import Path

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
        self.free_e = None # Free energy of adsorption

        self.pymol = Path(self.settings['pymol_dir']) # path to pymol installation
        self.packmol = Path(self.settings['packmol_dir']) # path to packmol installation
        self.acpype = Path(self.settings['acpype_dir']) # Path to acpype conda env

        subprocess.run("module load python/anaconda-2022.05  openmpi/4.1.1 gcc/10.2.0 cuda/11.2 fftw3/3.3.9 gsl/2.7 lapack/3.10.0", shell=True) # load required modules
    
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
            # get last frame of em
            subprocess.run(f"bash {self.script_dir}/get_last_frame.sh -f ./em.trr -s ./em.tpr -o ./PCC_em.pdb", check=True, shell=True)

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
            subprocess.run(f"module load gcc/10.2.0 && {self.packmol} < ./mix.inp", shell=True, check=True)
            # create topol.top and complex.itp
            top = GMXitp("./MOL.itp", "./PCC.itp")
            top.create_topol()
        ## EM
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
            # update atom ids
            self._get_atom_ids("./em.gro")
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
            subprocess.run(f"cp {self.script_dir}/nvt.mdp .", shell=True, check=True)
            # submit nvt job
            wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_nvt.sh", check=True, shell=True)
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
            subprocess.run(f"cp {self.script_dir}/npt.mdp .", shell=True, check=True)
            # submit npt job
            wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_complex_npt.sh npt", shell=True)
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
            # copy files into complex/pbmetad
            subprocess.run("cp ../MOL_truncated.itp .", shell=True, check=True)
            subprocess.run("cp ../em/posre_MOL.itp .", shell=True, check=True)
            subprocess.run("cp ../PCC_truncated.itp .", shell=True, check=True)
            subprocess.run("cp ../em/posre_PCC.itp .", shell=True, check=True)
            subprocess.run("cp ../complex.itp .", shell=True, check=True)
            subprocess.run("cp ../npt/topol.top .", shell=True, check=True)
            subprocess.run(f"cp {self.script_dir}/sub_mdrun_plumed.sh .", shell=True) # copy mdrun submission script
            subprocess.run(f"cp {self.script_dir}/plumed.dat ./plumed_temp.dat", shell=True) # copy pbmetad script
            # update PCC and MOL atom ids
            self._create_plumed("./plumed_temp.dat", "./plumed.dat")
            # remove temp plumed file
            subprocess.run(f"rm ./plumed_temp.dat", shell=True)
            # copy nvt.mdp into pbmetad
            subprocess.run(f"cp {self.script_dir}/md.mdp .", shell=True, check=True)
            # submit pbmetad job
            wait_str = " --wait " if wait else "" # whether to wait for pbmetad to finish before exiting
            subprocess.run(f"sbatch -J {self.PCC_code}{wait_str}sub_mdrun_plumed.sh", check=True, shell=True)
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
        return None
    
    def _calc_FE(self) -> None: # TODO
        return None
    
    def get_FE(self) -> float:
        """Wrapper for FE caclculations. Create PCC, call acpype, 
        minimize, create complex, and run PBMetaD.

        Returns:
            self.free_e (float): Free energy of adsorption b/w MOL and PCC
        """
        # create PCC
        #self._create_pcc()
        # get params
        #self._get_params()
        # minimize PCC
        #self._minimize_PCC()
        # create and minimize the complex
        #self._mix()
        # run PBMetaD
        #self._pbmetaD()
        # reweight
        self._reweight()
        # calc FE
        #self._calc_FE()
        return self.free_e