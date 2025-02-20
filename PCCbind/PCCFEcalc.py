import os
import json
import re
from pathlib import Path
import subprocess
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import MDAnalysis as md
from MDAnalysis.analysis import align
import numpy as np
from scipy.integrate import simpson
import pandas as pd

from GMXitp_PCC.GMXitp import GMXitp

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


class PCCFEcalc():

	def __init__(self, pcc1, pcc2, run_dir, settings_json):
		self.PCC1 = pcc1
		self.PCC2 = pcc2
		self.free_e = None
		self.free_e_err = None
		self.K = None
		self.K_err = None
		now = datetime.now()
		now = now.strftime("%m/%d/%Y, %H:%M:%S")
		print(f"{now}: Free energy calculations for {self.PCC1} with {self.PCC2}")
		self.base_dir = Path(run_dir)

		with open(settings_json) as f:
			self.settings = json.load(f)

		self.complex_dir = self.base_dir/"complex"
		if not self.complex_dir.exists():
			self.complex_dir.mkdir()
		
		# if the PCC calculations do not exist, look for them in the provided directory
		if not (self.base_dir/self.PCC1).exists():
			PCC1_calc_dir = self.settings.get("PCC_source", None)
			if PCC1_calc_dir is None:
				raise RuntimeError(f"Structure calculations source not found.")
			self.PCC1_calc_dir = Path(PCC1_calc_dir)/self.PCC1
			if not PCC1_calc_dir.exists():
				raise RuntimeError(f"Structure calculations for {self.PCC1} not found.")
		else:
			self.PCC1_calc_dir = self.base_dir/self.PCC1

			
		if not (self.base_dir/self.PCC2).exists():
			PCC2_calc_dir = self.settings.get("PCC_source", None)
			if PCC2_calc_dir is None:
				raise RuntimeError(f"Structure calculations source not found.")
			self.PCC2_calc_dir = Path(PCC2_calc_dir)/self.PCC2
			if not PCC2_calc_dir.exists():
				raise RuntimeError(f"Structure calculations for {self.PCC2} not found.")
		else:
			self.PCC2_calc_dir = self.base_dir/self.PCC2

	
		self.AAdict31 = {
			'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
            'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
			} # 3 to 1 translator
		self.AAdict13 = {j: i for i, j in self.AAdict31.items()} # 1 to 3 translator
		self.charge_dict = {"D": -1, "E": -1, "R": +1, "K": +1} # AA charges at neutral pH
		self.sys_charge = sum([self.charge_dict.get(i, 0) for i in list(self.PCC1)+list(self.PCC2)])
		
	
		self.mold_dir = Path(__file__).parent/Path("mold")
		self.num_umbrellas = int(self.settings["num_umbrellas"])
		self.max_dist = float(self.settings["max_distance"])
		self.centers = np.linspace(start=0.5, stop=self.max_dist, num=self.num_umbrellas)
		self.springK = float(self.settings["spring_constant"])
		self.T = float(self.settings["T"])
		self.KbT = self.T * 8.314 # System temperature in J/mol
		self.run_partition = self.settings['partition']
		self.accl = self.settings['accl']

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
		PCC1_list_id = []
		PCC1_list_atom = []
		PCC2_list_id = []
		PCC2_list_atom = []
		# get atom ids
		for line in atom_list:
			line_list = line.split()
			if re.match("^\d+PC1$", line_list[0]):
				PCC1_list_id.append(int(line_list[2]))
				PCC1_list_atom.append(line_list[1])
			elif re.match("^\d+PC2$", line_list[0]):
				PCC2_list_id.append(int(line_list[2]))
				PCC2_list_atom.append(line_list[1])
		# save PCC1_list and PCC2_list
		self.PCC1_list = PCC1_list_id
		self.PCC2_list = PCC2_list_id
		self.PCC1_list_atom = PCC1_list_atom
		self.PCC2_list_atom = PCC2_list_atom
		return None

	def _align(self) -> None:
		"""
		Align the two PCC and then shift them along the normal axis until there's no contact.
		"""

		# define helper functions
		def get_normal_vector(pcc: md.Universe) -> np.array:
			"""
			calculate the vector normal to the surface of the PCC. NOTE: THIS IS A STATIC
			METHOD. IF THE STRUCTURE OF THE PCC BACKBONE OR THE ATOM NUMBERINGS CHANGES,
			THIS WILL BE INCORRECT.
			"""
			a1 = pcc.atoms[4].position ## STATIC
			a2 = pcc.atoms[3].position ## STATIC
			a3 = pcc.atoms[7].position ## STATIC
			v1 = a2-a1
			v2 = a3-a1
			z_vec = np.cross(v2, v1)
			return z_vec/np.linalg.norm(z_vec)

		def check_contact(pcc1: md.Universe, pcc2: md.Universe, tol: float = 5) -> bool:
			"""
			Check for contact between pcc1 and pcc2 by caculating the pairwise distance
			between all atom pairs and checking against tol. Returns True if there is
			at least one distance < tol.
			"""
			coord1 = pcc1.coord.positions
			coord2 = pcc2.coord.positions
			pairds = []
			for atom1 in coord1:
				for atom2 in coord2:
					pairds.append(np.linalg.norm(atom2-atom1))
			if min(pairds) < tol:
				return True
			else:
				return False

		def shift_PCC(anchor: md.Universe, target: md.Universe, shift_dist):
			"""
			Shift target along the the normal vector to the anchor by shift_dist.
			Overwrites the previous coordinates of target.
			"""
			shift_vec = get_normal_vector(anchor)
			for atom in target.atoms:
				atom.position += shift_dist*shift_vec  
			
		# load PCC1
		PCC1 = md.Universe("./PCC1.pdb")
		PCC1.add_TopologyAttr("chainIDs")
		for atom in PCC1.atoms:
			atom.chainID = "A"
			atom.residue.resname = "PC1"
		
		# load PCC2
		PCC2 = md.Universe("./PCC2.pdb")
		PCC2.add_TopologyAttr("chainIDs")
		for atom in PCC2.atoms:
			atom.chainID = "B"
			atom.residue.resname = "PC2"
		
		# align the backbone of the PCCs. Residues are NOT aligned
		rmsds = align.alignto(PCC1.atoms[:38], PCC2.atoms[:38], match_atoms=True)
		
		# shift the aligned structure of PCC2 by 5 angstroms until there are no contacts between the structures.
		shift_increment = 5
		total_shift = 0
		
		while(check_contact(PCC1, PCC2)):
			shift_PCC(PCC1, PCC2, shift_increment)
			total_shift += shift_increment
		print("PCC total shift: ", total_shift)
		
		# merge the two structures and write complex.pdb
		complex_aligned = md.Merge(PCC1.atoms, PCC2.atoms)
		complex_aligned.atoms.write("./complex.pdb")	
		
	
	def _mix(self) -> None:
		"""
		Create the simulation box with MOL and PCC, and create initial structures for the PCC.
		
		Returns:
		    None
		"""
		if not self._check_done(self.complex_dir):
			## CREATE TOPOL.TOP
			with cd(self.complex_dir): # cd into complex
				# copy PCC files into complex directory
				subprocess.run(f"cp {self.PCC1_calc_dir}/PCC.acpype/PCC_GMX.itp ./PCC1.itp", shell=True, check=True)
				subprocess.run(f"cp {self.PCC1_calc_dir}/PCC.acpype/posre_PCC.itp ./posre_PCC1.itp", shell=True, check=True)
				subprocess.run(f"cp {self.PCC1_calc_dir}/em/PCC_em.pdb ./PCC1.pdb", shell=True, check=True)
				subprocess.run(f"cp {self.PCC2_calc_dir}/PCC.acpype/PCC_GMX.itp ./PCC2.itp", shell=True, check=True)
				subprocess.run(f"cp {self.PCC2_calc_dir}/PCC.acpype/posre_PCC.itp ./posre_PCC2.itp", shell=True, check=True)
				subprocess.run(f"cp {self.PCC2_calc_dir}/em/PCC_em.pdb ./PCC2.pdb", shell=True, check=True)
				# create complex.pdb with MDAnalysis
				self._align()
				# create topol.top and complex.itp
				top = GMXitp("./PCC1.itp", "./PCC2.itp")
				top.create_topol()
			
			self._set_done(self.complex_dir)
		return None

	def _eq_complex(self, wait: bool = True) -> None:
		"""
		Solvate, and equilibrate the complex.

		Args:
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
				subprocess.run("cp ../PCC1_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC1.itp .", shell=True, check=True)
				subprocess.run("cp ../PCC2_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC2.itp .", shell=True, check=True)
				subprocess.run("cp ../complex.itp .", shell=True, check=True)
				subprocess.run("cp ../complex.pdb .", shell=True, check=True)
				subprocess.run("cp ../topol.top .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/em/ions.mdp .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/em/em.mdp .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/em/sub_mdrun_complex_em_{self.run_partition}_cpu.sh ./sub_mdrun_complex_em.sh", shell=True) # copy mdrun submission script
				wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
				subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2}{wait_str}sub_mdrun_complex_em.sh {self.sys_charge}", check=True, shell=True)
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
				subprocess.run("cp ../PCC1_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../PCC2_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../complex.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC1.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC2.itp .", shell=True, check=True)
				subprocess.run("cp ../em/topol.top .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/nvt/sub_mdrun_complex_nvt_{self.run_partition}_{self.accl}.sh ./sub_mdrun_complex_nvt.sh", shell=True) # copy mdrun submission script
				# copy nvt.mdp into nvt
				if self.sys_charge != 0:
					subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt.mdp .", shell=True, check=True)
				else:
					subprocess.run(f"cp {self.mold_dir}/complex/nvt/nvt_nions.mdp ./nvt.mdp", shell=True, check=True)
				# submit nvt job
				wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
				subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2}{wait_str}sub_mdrun_complex_nvt.sh", check=True, shell=True)
			self._set_done(self.complex_dir/'nvt')
		## NPT
		if not self._check_done(self.complex_dir/"npt"):
			# create complex/npt dir
			Path.mkdir(self.complex_dir/"npt", exist_ok=True)
			with cd(self.complex_dir/"npt"): # cd into complex/npt
				# copy files into complex/npt
				subprocess.run("cp ../PCC1_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC1.itp .", shell=True, check=True)
				subprocess.run("cp ../PCC2_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC2.itp .", shell=True, check=True)
				subprocess.run("cp ../complex.itp .", shell=True, check=True)
				subprocess.run("cp ../nvt/topol.top .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/npt/sub_mdrun_complex_npt_{self.run_partition}_{self.accl}.sh ./sub_mdrun_complex_npt.sh", shell=True) # copy mdrun submission script
				# copy npt.mdp into npt
				if self.sys_charge != 0:
					subprocess.run(f"cp {self.mold_dir}/complex/npt/npt.mdp .", shell=True, check=True)
				else:
					subprocess.run(f"cp {self.mold_dir}/complex/npt/npt_nions.mdp ./npt.mdp", shell=True, check=True)
				# submit npt job
				wait_str = " --wait " if wait else "" # whether to wait for em to finish before exiting
				subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2}{wait_str}sub_mdrun_complex_npt.sh", shell=True)
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
	

	def _create_plumed_us(self, plumed_in: Path, plumed_out: Path, usample_center) -> None:
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
		assert self._is_continuous(self.PCC1_list), "PCC1 id list is not continuous. Check complex/em/em.gro."
		assert self._is_continuous(self.PCC2_list), "PCC2 id list is not continuous. Check complex/em/em.gro."
		# define atom ranges for PCC and MOL
		PCC1_atom_id = f"{min(self.PCC1_list)}-{max(self.PCC1_list)}"
		PCC2_atom_id = f"{min(self.PCC2_list)}-{max(self.PCC2_list)}"
		# replace new ids
		for i, line in enumerate(cnt):
			if "$1" in line:
				line = line.replace("$1", PCC1_atom_id)
			if "$2" in line:
				line = line.replace("$2", PCC2_atom_id)
			if "$3" in line:
				line = line.replace("$3", str(usample_center))
			
			cnt[i] = line
		# write new plumed file
		with open(plumed_out, 'w') as f:
			f.writelines(cnt)

		return None
	
	def _create_plumed(self, plumed_in: Path, plumed_out: Path) -> None:
		"""
		Fix the plumed tempelate with PCC1 and PCC2 atom ids. DOES NOT SUPPORT NON-CONTINUOUS ATOM IDS.

		Args:
			plumed_in (Path): Path to input plumed file
			plumed_out (Path): Path to output plumed file

		Raises:
			AssertionError: If `self.PCC1_list` or `self.PCC2_list` are not continuous.
			
		Returns:
			None
		"""
		# read plumed files
		with open(plumed_in) as f:
			cnt = f.readlines()
		# make sure id lists are continuous
		assert self._is_continuous(self.PCC1_list), "MOL id list is not continuous. Check complex/em/em.gro."
		assert self._is_continuous(self.PCC2_list), "PCC id list is not continuous. Check complex/em/em.gro."
		# define atom ranges for PCC and MOL
		PCC1_atom_id = f"{min(self.PCC1_list)}-{max(self.PCC1_list)}"
		PCC2_atom_id = f"{min(self.PCC2_list)}-{max(self.PCC2_list)}"
		p1a_list = [self.PCC1_list[self.PCC1_list_atom.index(i)] for i in ["N4", "C10", "C11", "O1"]]
		p1b_list = [self.PCC1_list[self.PCC1_list_atom.index(i)] for i in ["C1", "C2", "C3", "O"]]
		vp1a_atom_ids = "".join([f"{i}," for i in p1a_list])[:-1]
		vp1b_atom_ids = "".join([f"{i}," for i in p1b_list])[:-1]
		vr1b_list = [self.PCC1_list[self.PCC1_list_atom.index(i)] for i in ["N1", "N2", "N3", "C7", "C8"]]
		vr1b_atom_ids = "".join([f"{i}," for i in vr1b_list])[:-1]
		p2a_list = [self.PCC2_list[self.PCC2_list_atom.index(i)] for i in ["N4", "C10", "C11", "O1"]]
		p2b_list = [self.PCC2_list[self.PCC2_list_atom.index(i)] for i in ["C1", "C2", "C3", "O"]]
		vp2a_atom_ids = "".join([f"{i}," for i in p2a_list])[:-1]
		vp2b_atom_ids = "".join([f"{i}," for i in p2b_list])[:-1]
		vr2b_list = [self.PCC2_list[self.PCC2_list_atom.index(i)] for i in ["N1", "N2", "N3", "C7", "C8"]]
		vr2b_atom_ids = "".join([f"{i}," for i in vr2b_list])[:-1]
		
		# replace new ids
		for i, line in enumerate(cnt):
			if "$1" in line:
				line = line.replace("$1", PCC1_atom_id)
			if "$2" in line:
				line = line.replace("$2", PCC2_atom_id)
			if "$3" in line:
				line = line.replace("$3", vp1a_atom_ids)
			if "$4" in line:
				line = line.replace("$4", vp1b_atom_ids)
			if "$5" in line:
				line = line.replace("$5", vp2a_atom_ids)
			if "$6" in line:
				line = line.replace("$6", vp2b_atom_ids)
			if "$7" in line:
				line = line.replace("$7", vr1b_atom_ids)
			if "$8" in line:
				line = line.replace("$8", vr2b_atom_ids)
			
			cnt[i] = line
		# write new plumed file
		with open(plumed_out, 'w') as f:
			f.writelines(cnt)

		return None
	
	def _get_slurm_report(self) -> list:
			slurm = subprocess.check_output("""sacct --format="JobID%20,JobName%30" -s R,PD""", shell=True)
			slurm = str(slurm).split("\\n")
			running = []
			for i in slurm[2:-1]:
				item = i.split()
				if re.match(f"^{self.PCC1}\_{self.PCC2}$", item[-1]):
					ilist = item[0].split("_")
					if ilist[-1][0] == "[": # pending jobs are concantenated in slurm. Have to handle them
						# the dirty way...
						run_range = list(map(int, ilist[-1].strip("[]").split("-")))
						for j in range(run_range[0], run_range[1]+1):
							if j not in running:
								running.append(j)
					else:
						running.append(int(ilist[-1]))
			return running
	
	def _check_done_usample(self) -> None:
		runs_dir = self.complex_dir/"usample"/"runs"
		run_list = [str(i) for i in range(self.num_umbrellas)]
		timeout = 144
		loop_cnt = 0
		while True:
			if loop_cnt > timeout:
				raise TimeoutError("Timeout (wait > 2d).")
			running = self._get_slurm_report()
			if len(run_list) == 0: # if there are no running sims, just exit
				return None
			for i in run_list:
				if (runs_dir/i/"md2.gro").exists(): # Check if completed
					self._set_done(runs_dir/i)
					run_list.remove(i) # remove from running list
					loop_cnt -= 1
					break # break out of the for loop so the realtime removing of 
					#the elements doesn't mess up the work flow.
				elif int(i) not in running: # raise if not in the queue
					raise RuntimeError(f"Run {i} is no longer running, possibly failed. Check the run directory.")
				else: # if it's still running, wait 10 minutes, restart the while loop
					time.sleep(60*10)
					break
			loop_cnt += 1
	
	def _usample(self) -> None:
		"""
		Run usample from equilibrated structure.

		Args:
			wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

		Returns:
			None
		"""
		# create complex/usample dir
		Path.mkdir(self.complex_dir/"usample", exist_ok=True)
		with cd(self.complex_dir/"usample"): # cd into complex/usample
			if not Path("./.setupdone").exists():
				subprocess.run(f"cp {self.mold_dir}/complex/usample/sub_usample_array_{self.run_partition}_{self.accl}.sh ./sub_usample.sh", shell=True) # copy mdrun submission script
				Path.mkdir(Path("./runs"), exist_ok=True) # create runs dir
				for i in range(self.num_umbrellas): # create separate run folders
					Path.mkdir(Path(f"./runs/{i}"), exist_ok=True)
				# copy files into complex/usample
				for i in range(self.num_umbrellas):
					with cd(f"./runs/{i}"):
						if self._check_done(Path(f"complex/usample/runs/{i}")):
							continue
						subprocess.run("cp ../../../PCC1_truncated.itp .", shell=True, check=True)
						subprocess.run("cp ../../../posre_PCC1.itp .", shell=True, check=True)
						subprocess.run("cp ../../../PCC2_truncated.itp .", shell=True, check=True)
						subprocess.run("cp ../../../posre_PCC2.itp .", shell=True, check=True)
						subprocess.run("cp ../../../complex.itp .", shell=True, check=True)
						subprocess.run("cp ../../../npt/topol.top .", shell=True, check=True)
						subprocess.run(f"cp {self.mold_dir}/complex/usample/plumed.dat ./plumed_temp.dat", shell=True) # copy pbmetad script
						# update PCC atom ids
						self._create_plumed("./plumed_temp.dat", "./plumed.dat", self.centers[i])
							# remove temp plumed file
						subprocess.run(f"rm ./plumed_temp.dat", shell=True)
						# copy md1/2.mdp into rundirs
						if self.sys_charge != 0:
							subprocess.run(f"cp {self.mold_dir}/complex/usample/md1.mdp .", shell=True, check=True)
							subprocess.run(f"cp {self.mold_dir}/complex/usample/md2.mdp .", shell=True, check=True)
						else:
							subprocess.run(f"cp {self.mold_dir}/complex/usample/md1_nions.mdp ./md1.mdp", shell=True, check=True)
							subprocess.run(f"cp {self.mold_dir}/complex/usample/md2_nions.mdp ./md2.mdp", shell=True, check=True)
				with open(Path("./.setupdone"), 'w') as f:
					f.write("")
				
			running = self._get_slurm_report()
			if len(running) == 0:
				# failsafe: check for already completed runs
				completed = []
				for i in range(self.num_umbrellas):
					if Path(f"./runs/{i}/md2.gro").exists():
						completed.append(i)
						with open(Path(f"./runs/{i}/.done"), 'w') as f:
							f.write("")
				if len(completed) != self.num_umbrellas:
					subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2} sub_usample.sh", check=True, shell=True)
		time.sleep(10)
		self._check_done_usample()
		self._set_done(self.complex_dir/'usample')
		return None
	
	def _pbmetad(self):
		"""
		Run PBMetaD from equilibrated structure.

		Args:
			rot_key: (int): Which side of the PCC to run.
			wait (bool, optional): Whether or not to wait for the sims to finish. Defaults to True.

		Returns:
			None
		"""
		# create complex/pbmetad dir
		Path.mkdir(self.complex_dir/"pbmetad", exist_ok=True)
		with cd(self.complex_dir/"pbmetad"): # cd into complex/pbmetad
			wait_str = "--wait"
			if Path.exists(self.complex_dir/"pbmetad"/"md.cpt"): # if there's a checkpoint, continue the run
				now = datetime.now()
				now = now.strftime("%m/%d/%Y, %H:%M:%S")
				print(f"{now}: Resuming previous run...", flush=True)
				subprocess.run("mv ./HILLS_ang ./HILLS_ang.bck.unk", shell=True, check=False)
				subprocess.run("mv ./HILLS_cos ./HILLS_cos.bck.unk", shell=True, check=False)
				subprocess.run("mv ./HILLS_COM ./HILLS_COM.bck.unk", shell=True, check=False)
				subprocess.run("cp ./md.cpt ./md.cpt.bck.unk", shell=True, check=True)
			else:
				# copy files into complex/pbmetad
				subprocess.run("cp ../PCC1_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC1.itp .", shell=True, check=True)
				subprocess.run("cp ../PCC2_truncated.itp .", shell=True, check=True)
				subprocess.run("cp ../posre_PCC2.itp .", shell=True, check=True)
				subprocess.run("cp ../complex.itp .", shell=True, check=True)
				subprocess.run("cp ../npt/topol.top .", shell=True, check=True)
				subprocess.run(f"cp {self.mold_dir}/complex/pbmetad/sub_mdrun_plumed.sh .", shell=True) # copy mdrun submission script
				subprocess.run(f"cp {self.mold_dir}/complex/pbmetad/plumed.dat ./plumed_temp.dat", shell=True) # copy pbmetad script
				subprocess.run(f"cp {self.mold_dir}/complex/pbmetad/plumed_restart.dat ./plumed_r_temp.dat", shell=True) # copy pbmetad script
				# update PCC1 and PCC2 atom ids
				self._create_plumed("./plumed_temp.dat", "./plumed.dat")
				self._create_plumed("./plumed_r_temp.dat", "./plumed_restart.dat")
				# remove temp plumed file
				subprocess.run(f"rm ./plumed_temp.dat", shell=True)
				subprocess.run(f"rm ./plumed_r_temp.dat", shell=True)
				# copy md.mdp into pbmetad
				if self.sys_charge != 0:
					subprocess.run(f"cp {self.mold_dir}/complex/pbmetad/md.mdp .", shell=True, check=True)
				else:
					subprocess.run(f"cp {self.mold_dir}/complex/pbmetad/md_nions.mdp ./md.mdp", shell=True, check=True)
			# submit pbmetad job. Resubmits until either the job fails 10 times or it succesfully finishes.
			cnt = 1
			try:
				subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2} {wait_str} sub_mdrun_plumed.sh", check=True, shell=True)
				if not Path.exists(self.complex_dir/"pbmetad"/"md.gro"): # making sure except block is executed if the run is not complete, regardless of system exit code
					raise RuntimeError("Run not completed.")
			except:
				fail_flag = True
				while fail_flag:
					try:
						cnt += 1
						subprocess.run(f"mv ./HILLS_ang ./HILLS_ang.bck.{cnt}", shell=True, check=False)
						subprocess.run(f"mv ./HILLS_cos ./HILLS_cos.bck.{cnt}", shell=True, check=False)
						subprocess.run(f"mv ./HILLS_COM ./HILLS_COM.bck.{cnt}", shell=True, check=False)
						subprocess.run(f"cp ./md.cpt ./md.cpt.bck.{cnt}", shell=True, check=True)
						now = datetime.now()
						now = now.strftime("%m/%d/%Y, %H:%M:%S")
						print(f"{now}: Resubmitting PBMetaD: ", end="", flush=True)
						subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2} {wait_str} sub_mdrun_plumed.sh", check=True, shell=True)
						print()
						fail_flag = False
					except:
						if cnt >= 10:
							raise RuntimeError("PBMetaD run failed more than 10 times. Stopping.")

		self._set_done(self.complex_dir/'pbmetad')
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
			subprocess.run("cp ../pbmetad/GRID_COM .", shell=True, check=True)
			subprocess.run("cp ../pbmetad/GRID_ang .", shell=True, check=True)
			subprocess.run("cp ../pbmetad/GRID_cos .", shell=True, check=True)
			subprocess.run(f"cp {self.mold_dir}/complex/reweight/sub_mdrun_rerun.sh .", shell=True) # copy mdrun submission script            
			subprocess.run(f"cp {self.mold_dir}/complex/reweight/reweight.dat ./reweight_temp.dat", shell=True) # copy reweight script
			# update PCC and MOL atom ids
			self._create_plumed("./reweight_temp.dat", "./reweight.dat")
			# remove temp plumed file
			subprocess.run(f"rm ./reweight_temp.dat", shell=True)
			# submit reweight job
			wait_str = " --wait " if wait else "" # whether to wait for reweight to finish before exiting
			subprocess.run(f"sbatch -J {self.PCC1}_{self.PCC2} {wait_str} sub_mdrun_rerun.sh", check=True, shell=True)
		self._set_done(self.complex_dir/'reweight')
		return None

	def _create_metadata_rust(self, fdir):
		metadata_cnt = []

		for i, dist in enumerate(self.centers):
			trj_path = self.complex_dir/"usample"/"runs"/f"{i}"/"us.dat"
			metadata_cnt.append(f"{trj_path}\t{dist:.6f}\t{self.springK}\n")

		with open(fdir, 'w') as f:
			f.writelines(metadata_cnt)

	def _get_corr_time(self, fname):
		with open(fname) as f:
			line = f.readline()
			cnt = []
			while line:
				if line[0] == "#":
					line = f.readline()
					continue
				else:
					line = line.strip().split()
					cnt.append(float(line[1]))
					line = f.readline()
		cnt = np.array(cnt)
		acf = np.array([1]+[np.corrcoef(cnt[:-i], cnt[i:])[0,1]  \
				for i in range(1, 20000)])
		autocorr_time = 2 * (np.where(acf < 1/np.e)[0][0]-1)
		return autocorr_time
        
	def _create_metadata(self, fdir):
		metadata_cnt = []
		with open(self.complex_dir/"usample"/"runs"/"corr_time.json") as f:
			corr_time = json.load(f)
		for i, dist in enumerate(self.centers):
			trj_path = self.complex_dir/"usample"/"runs"/f"{i}"/"us.dat"
			corr_time_i = (float(corr_time[str(i)])//100 + 1) * 100 # rounding up to the closest 100
			metadata_cnt.append("\t".join(list(map(str, [trj_path, dist, self.springK, corr_time_i, "\n"]))))

		with open(fdir, 'w') as f:
				f.writelines(metadata_cnt)

	def _wham(self, wait: bool = True) -> None:
		"""
		create the metadata file and submit the wham reweighting job
		"""
		wait_str = " --wait " if wait else "" # whether to wait for wham to finish before exiting the script
		# create complex/wham_dir
		Path.mkdir(self.complex_dir/"wham", exist_ok=True)
		# check if metadata exists:
		if not (self.complex_dir/"usample"/"runs"/"metadata.dat").exists():
			if not (self.complex_dir/"usample"/"runs"/"corr_time.json").exists():
				with cd(self.complex_dir/"wham"):
					subprocess.run(f"cp {self.mold_dir}/complex/wham/sub_calc_corrtime.sh .", shell=True) # copy corr_time submission script
					subprocess.run(f"cp {self.mold_dir}/complex/wham/calc_corrtime.py .", shell=True) # copy corr_time script
					subprocess.run(f"sbatch -J corr_{self.PCC1}_{self.PCC2}{wait_str}sub_calc_corrtime.sh {self.complex_dir}/usample/runs {self.num_umbrellas}", check=True, shell=True)
			self._create_metadata(self.complex_dir/"usample"/"runs"/"metadata.dat")
					
		with cd(self.complex_dir/"wham"): # cd into complex/wham
			subprocess.run(f"cp {self.mold_dir}/complex/wham/sub_wham_{self.run_partition}_cpu.sh ./sub_wham.sh", shell=True) # copy wham submission script
			subprocess.run(f"sbatch -J wham_{self.PCC1}_{self.PCC2}{wait_str}sub_wham.sh {self.max_dist+0.2}", check=True, shell=True) # run wham
		self._set_done(self.complex_dir/"wham")

	def _read_wham(self, fname: Path) -> pd.DataFrame:
		data = pd.DataFrame([], columns=["X", "F", "err"])
		with open(fname) as f:
			line = f.readline()
			while line:
				if line[0] == "#":
					line = f.readline()
					continue
				line = line.split()
				line_df = pd.DataFrame({"X": [float(line[0])], 
							"F": [float(line[1])], 
							"err": [float(line[2])]})
				data = pd.concat([data if not data.empty else None, line_df], ignore_index=True)
				line = f.readline()
		return data
	
	def _write_report(self) -> None:
		report = {
            "PCC1": self.PCC1,
            "PCC2": self.PCC2,
            "FE": self.free_e,
            "FE_error": self.free_e_err,
			"K": self.K,
			"K_error": self.K_err
        }
		with open(self.base_dir/"metadata.JSON", 'w') as f:
			json.dump(report, f, indent=3)
		return None
	
	def _calc_I(self, data: pd.DataFrame) -> float:
		"""
		Calculate the area under pmf in `data` using scipy.simson.

		Args:
			data (pd.DataFrame): DataFrame with columns x(dcom), F(free energy), err(error)

		Returns:
			float: Simpson integrand
		"""
		data["exp"] = np.exp(-data.F*1000/self.KbT)
		# integrate over X
		data.sort_values(by='X', inplace=True)
		integrand = simpson(y=data.exp.to_numpy(), x=data.X.to_numpy())
		return integrand

	
	def _calc_I_err_sq(self, data: pd.DataFrame) -> float:
		"""
		Calculate squared error of the free energy integrated with scipy.simpson with uniform intervals.
		data = DataFrame with columns x(dcom), F(free energy), err(error)
		"""
		# f = exp(-F/kbT)
		# (df)**2 = (df/dF)**2 * (dF)**2
		dfdF = (-1/self.KbT) * np.exp(-data.F.to_numpy()*1000/self.KbT) * (data.err.to_numpy() * 1000)
		d = self.centers[1]-self.centers[0]
		I_err_sq = dfdF[0]**2 # i = 0
		if dfdF.squeeze().shape[0]%2 == 0: # odd intervals
			for i, err in enumerate(dfdF[1:-3]):
				if (i+1)%2 == 0: # we took out the first item
					I_err_sq += 4*(err**2)
				else:
					I_err_sq += 16*(err**2)
			I_err_sq += 25/16 * dfdF[-3]**2 # i = N-2
			I_err_sq += 4 * dfdF[-2]**2 # i = N-1
			I_err_sq += 1/16 * dfdF[-1]**2 # i = N
		else: # even intervals
			for i, err in enumerate(dfdF[1:]):
				if (i+1)%2 == 0:
					I_err_sq += 4*(err**2)
				else:
					I_err_sq += 16*(err**2)
		I_err_sq *= d**2/9
		return I_err_sq
	
	def _calc_deltaF_usample(self, bound_data: pd.DataFrame, unbound_data: pd.DataFrame) -> tuple:
		"""
		Calculate the free energy difference between bound and unbound states and it's error.

		Args:
			bound_data (pd.DataFrame): DataFrame of bound state with columns X(dcom), F(free energy), err(error)
			unbound_data (pd.DataFrame): DataFrame of unbound state with columns X(dcom), F(free energy), err(error)

		Returns:
			Tuple: (\Delta F, \sigma \Delta F)
		"""
		r_int = self._calc_I(bound_data.copy())
		p_int = self._calc_I(unbound_data.copy())
		r_F = -self.KbT*np.log(r_int)/1000
		p_F = -self.KbT*np.log(p_int)/1000
		dF = r_F - p_F

		r_dIsq = self._calc_I_err_sq(bound_data.copy())
		r_errsq = ((-self.KbT/r_int)**2) * r_dIsq
		p_dIsq = self._calc_I_err_sq(unbound_data.copy())
		p_errsq = ((-self.KbT/p_int)**2) * p_dIsq
		ddF = np.sqrt(r_errsq + p_errsq)/1000
		return dF, ddF
	
	def _calc_FE_usample(self) -> tuple:
		# bound = 0.5<=dcom<=2.0 nm
		wham_data = self._read_wham(self.complex_dir/"wham"/"wham.out")
		wham_data.F = wham_data.F.replace(np.inf, np.nan)
		wham_data.dropna(inplace=True)
		wham_data.F += self.KbT*2*np.log(wham_data.X)/1000
		bound_data = wham_data[(wham_data.X>=0.5) & (wham_data.X<=2.0)]
		#bound_data.dropna(inplace=True)
		# unbound = 2.0<dcom<4.0~inf nm 
		unbound_data = wham_data[(wham_data.X>2.0) & (wham_data.X<self.max_dist)]
		#unbound_data.dropna(inplace=True)
		self.free_e, self.free_e_err = self._calc_deltaF(bound_data=bound_data, unbound_data=unbound_data)
		return self.free_e, self.free_e_err
	
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
		if len(data["time"]) < 4000000:
			raise RuntimeError("The simulation might not have completed correctly. Check the md folder.")
		data = pd.DataFrame(data)
		data['weights'] = np.exp(data['pb.bias']*1000/self.KbT)
		#init_time = self._find_converged() #ns
		init_time = 100 # ns
		print(f"INFO: Discarding initial {init_time} ns of data for free energy calculations.")
		if init_time > 300:
			raise RuntimeError("Large hill depositions detected past 300 ns mark. Check the convergence of the PBMetaD calculations.")
		init_idx = int(init_time * 10000 // 2)
		data = data.iloc[init_idx:] # discard the first 100 ns of data
		return data
	
	def _block_anal_3d(self, x, y, z, weights, block_size=None, folds=None, nbins=100):
		# calculate histogram for all data to get bins
		_, binsout = np.histogramdd([x, y, z], bins=nbins, weights=weights)
		# calculate bin centers
		binsx, binsy, binsz = binsout
		xs = np.round((binsx[1:] + binsx[:-1])/2, 2)
		ys = np.round((binsy[1:] + binsy[:-1])/2, 2)
		zs = np.round((binsz[1:] + binsz[:-1])/2, 2)
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
		zs_unrolled = []
		for i in xs:
			for j in ys:
				for k in zs:
					xs_unrolled.append(i)
					ys_unrolled.append(j)
					zs_unrolled.append(k)

		data['x'] = xs_unrolled
		data['y'] = ys_unrolled
		data['z'] = zs_unrolled

		# calculate free energy for each fold
		for fold in range(folds):
			x_fold = x[block_size*fold:(fold+1)*block_size]
			y_fold = y[block_size*fold:(fold+1)*block_size]
			z_fold = z[block_size*fold:(fold+1)*block_size]
			weights_fold = weights[block_size*fold:(fold+1)*block_size]
			counts, _ = np.histogramdd([x_fold, y_fold, z_fold], bins=[binsx, binsy, binsz], weights=weights_fold)
			counts[counts==0] = np.nan # discard empty bins
			free_energy = -self.KbT*np.log(counts)/1000 #kJ/mol
			free_energy_unrolled = []
			for i in range(len(xs)):
				for j in range(len(ys)):
					for k in range(len(zs)):
						free_energy_unrolled.append(free_energy[i, j, k])
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
		data = DataFrame with columns x(dcom), y(angle), z(cos), F(free energy)
		"""
		data["exp"] = np.exp(-data.F*1000/self.KbT)
		# integrate over Z
		Z_integrand = {"x": [], "y": [], "exp":[]}
		for x in data.x.unique():
			for y in data.y.unique():
				FE_this_xy = data[(data.x == x) & (data.y == y)].copy()
				FE_this_xy.sort_values(by='z', inplace=True)
				Z_integrand["x"].append(x)
				Z_integrand["y"].append(y)
				if FE_this_xy.empty: # if it doesn't exist, it's empty
					Z_integrand["exp"].append(0.0)
				else:
					Z_integrand["exp"].append(simpson(y=FE_this_xy.exp.to_numpy(), x=FE_this_xy.z.to_numpy()))

		Z_integrand_pd = pd.DataFrame(Z_integrand)
		# integrate over Y
		Y_integrand = {"x": [], "exp":[]}
		for x in Z_integrand_pd.x.unique():
			FE_this_x = Z_integrand_pd[Z_integrand_pd.x == x].copy()
			FE_this_x.sort_values(by='y', inplace=True)
			Y_integrand["x"].append(x)
			if FE_this_x.empty:
				Y_integrand["exp"].append(0.0)
			else:
				Y_integrand["exp"].append(simpson(y=FE_this_x.exp.to_numpy(), x=FE_this_x.y.to_numpy()))

		# integrate over X
		Y_integrand_pd = pd.DataFrame(Y_integrand)
		Y_integrand_pd.sort_values(by='x', inplace=True)
		integrand = simpson(y=Y_integrand_pd.exp.to_numpy(), x=Y_integrand_pd.x.to_numpy())

		return -self.KbT*np.log(integrand)/1000

	def _calc_deltaF(self, bound_data, unbound_data):
		r_int = self._calc_region_int(bound_data.copy())
		p_int = self._calc_region_int(unbound_data.copy())
		return r_int - p_int
	
	def _calc_FE(self) -> None:
		colvars = self._load_plumed() # read colvars
		# block analysis
		block_anal_data = self._block_anal_3d(colvars.dcom, colvars.ang, 
											colvars.dircos, colvars.weights, 
											nbins=50, block_size=5000*100)
		f_list = []
		f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
		discarded_blocks = 0
		for i in f_cols:
			try:
				# bound = 0<=dcom<=1.5 nm
				bound_data = block_anal_data[(block_anal_data.x>=0.0) & (block_anal_data.x<=2.0)][['x', 'y', 'z', i, 'ste']]
				bound_data.rename(columns={i: 'F'}, inplace=True)
				bound_data.dropna(inplace=True)
				# unbound = 2.0<dcom<2.4~inf nm 
				unbound_data = block_anal_data[(block_anal_data.x>2.0) & (block_anal_data.x<self.max_dist)][['x', 'y', 'z', i, 'ste']]
				unbound_data.rename(columns={i: 'F'}, inplace=True)
				unbound_data.dropna(inplace=True)
				f_list.append(self._calc_deltaF(bound_data=bound_data, unbound_data=unbound_data))
			except:
				discarded_blocks += 1
				continue

		if discarded_blocks != 0:
			print(f"WARNING: {discarded_blocks} block(s) were discarded from the calculations possibly because the system was"\
					" stuck in a bound state for longer than 100 ns consecutively. Check the colvar trajectories.")
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

	def run(self) -> None:
		"""
		Wrapper for PCCFECalc. Create complex box, equilibrate, run umbrella sampling runs, reweight with WHAM and calculate the 
		free energy of binding.
		"""
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
		# run usample
		now = datetime.now()
		now = now.strftime("%m/%d/%Y, %H:%M:%S")
		#print(f"{now}: Starting umbrella sampling runs: ", end="", flush=True)
		#if not self._check_done(self.complex_dir/'usample'):
		#	self._usample()
		#print("\tDone.")
		print(f"{now}: Starting PBmetaD: ", end="", flush=True)
		if not self._check_done(self.complex_dir/'pbmetad'):
			self._pbmetad()
		print("\tDone.")
		# reweight
		#now = datetime.now()
		#now = now.strftime("%m/%d/%Y, %H:%M:%S")
		#print(f"{now}: Running WHAM: ", end="", flush=True)
		#if not self._check_done(self.complex_dir/'wham'):
		#	self._wham()
		#print("\tDone.", flush=True)
		print(f"{now}: Reweighting: ", end="", flush=True)
		if not self._check_done(self.complex_dir/'reweight'):
			self._reweight()
		print("\tDone.")
		#postprocess
		now = datetime.now()
		now = now.strftime("%m/%d/%Y, %H:%M:%S")
		print(f"{now}: Postprocessing: ", flush=True)
		self._postprocess()
		now = datetime.now()
		now = now.strftime("%m/%d/%Y, %H:%M:%S")
		print(f"{now}: All steps completed.")
		print("-"*30 + "Finished" + "-"*30)
