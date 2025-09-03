# PCC-FECalc: Binding Free Energy Calculations for Protein-Catalyzed Capture Agents

## 📖 Overview

This repository provides a python framework for performing binding free energy calculations for a user-defined target molecule and a protein-catalyzed capture (PCC) agent with a user-defined epitope sequence. The core of this package is based on ["High-throughput virtual screening of protein-catalyzed capture agents for novel hydrogel-nanoparticle fentanyl sensors"](10.26434/chemrxiv-2025-psxft) and the corresponding [repository](https://github.com/Ferg-Lab/FEN-HTVS). Please refer to the original paper for a more technical explanation of the theory and the setup behind the free energy calculations done with this framework.

---

## 🚀 Features

The framework is organized into four main submodules that work in serial to perform a complete binding free energy calculation using GROMACS:

- `PCCBuilder`: This object creates a PCC with a given sequence, calculates GAFF2 parameters for it, and minimizes it.
- `TargetMOL`: Similar to `PCCBuilder`, creates input structures and force field parameters for a provided target molecule.
- `FECalc`: Brings together the PCC and the target molecule, solvates them in a water box, and performs minimization, NVT, and NPT equilibration, runs a parallel-bias metadynamics simulation, and reweights the resulting trajectory for free energy calculations.
- `postprocess`: Uses the reweighted statistics to calculate the 3D binding free energy volume of the PCC-molecule complex and calculates the binding free energy and dissociation constant.

---

## 📦 Installation

### Clone the repository

``` bash
    git clone https://github.com/arminshzd/PCC-FECalc.git
    cd PCC-FECalc
```

### Install

``` bash
pip install -e .
```

---

## 🛠 Usage

See `example/pcc_submit_test.py` for a sample calculation setup.

This framework additionally relies on PyMOL and `acpype` for PCC mutations and GAFF2 parameter generation. Both are installed automatically as the `pymol-open-source` and `acpype` Python packages. Ensure the `pymol` executable is available on your `PATH`.

The calculations happen through four steps and each step requires a `JSON` files with the necessary user parameters:

### Step 1: Building the PCC

The settings file for this step is pre-made in `FECalc/PCCBuilder_settings.JSON` and generally requires no modification.

### Step 2: Building the target

The settings file for this step should be created by the user. Two example are provided in `example/ACT_settings.JSON` for acetaminophen and `example/FEN_settings.JSON` for fentanyl. The mandatory entries are:

- `name`: Name of the target. Used for creating subdirectories and making reports.
- `charge`: Total charge of the target.
- `anchor1`: Anchor point defined using the atoms on the target molecule. This is used together with `anchor2` to define a vector that is used in determining the relative position and orientation of the target molecule with respect to the PCC during the PBMetaD calculations. See the original publication for a detailed explanation of how this vector is used in the collective variables.
- `anchor2`: See above.
- `output_dir`: Path to the folder to store the parameter calculations and minimization.
- `input_pdb_dir`: Input `pdb` file of the target molecule structure.

### Step 3: Enhanced sampling calculations

The settings file for this step should be created by the user. An example is provided in `example/system_settings.JSON`. The mandatory entries are:

- `PCC_output_dir`: Path to the output folder that holds the PCC calculations.
- `PCC_settings_json`: Path to the `JSON` file for the PCC.
- `MOL_settings_json`: Path to the `JSON` file for the target.
- `temperature`: Temperature of the simulations
- `box_size`: Size of the simulation box. Cubic periodic.
- `complex_output_dir`: Path to the out directory for the free energy calculations. The contents of this directory will be as follows:

``` bash
{PCC sequence}_{target name}/
│-- em/               # Minimization
│-- nvt/              # NVT equilibration
│-- npt/              # NPT equilibration
│-- md/               # PBMetaD simulation
│-- reweight/         # Reweighting
```

The optional entries are:

- `scheduler`: Scheduler used to allocate hardware resources. Supported
  options are `"local"` (default), `"slurm"`, `"pbs"`, and `"lsf"`.
- `nodes`, `cores`, `threads`: Hardware layout overrides for the number of
  nodes, cores per node, and threads per core. Defaults to ``1`` for each
  value; scheduler environment variables take precedence. These counts are
  used to set the correct `-ntomp` for GROMACS.
- `metad_settings`: Parameters of the metadynamics simulation
  - `n_steps`: Number of steps for the metadynamics run. 2 fs step size. defaults to 800 ns.
  - `metad_height`: Height of the deposited Guassians. Defaults to 3.0 kJ/mol.
  - `metad_pace`: Pace of deposition. Defaults to 500 steps.
  - `metad_bias_factor`: Biasing factor for the PBMetaD bias. Defaults to 20.
- `postprocess_settings`: Parameters for the post-processing and free energy calculations.
  - `discard_initial`: Initial duration of the PBMetaD simulation to discard for free energy calculations in ns. Defaults to 100 ns.
  - `n_folds`: Number of folds for block-analysis and uncertainty quantification. Defaults to 5.

### Step 4: Free energy calculation

This step does not require a separate settings file as the parameters are set by the settings file in step 3. The post-processing step calculates the 3D integral of the bound basin and the unbound region and reports $\Delta G$ of binding in kJ/mol and $K_d$ in $\mu M$. The report is created in `complex_output_dir/{PCC sequence}_{target name}/metadata.JSON`.

---

## 📜 License

This project is licensed under the MIT License.
