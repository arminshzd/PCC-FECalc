# Instructions

To install PCC-FECalc from source, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/arminshzd/PCC-FECalc.git
cd PCC-FECalc
pip install -e .
```

After installing the repository, add AmberTools and `libgfortran5` to enable the `sqm` program:

```bash
conda install -c conda-forge ambertools libgfortran5
```

> **Note:** PyMOL is required but not installed automatically because no pip wheel is available. Install PyMOL separately from [pymol.org](https://www.pymol.org/) or build from the open-source [GitHub repository](https://github.com/schrodinger/pymol-open-source). There's also a conda package package available:
>
> ```bash
> conda install -c conda-forge pymol-open-source
> ```
>
> Ensure the `pymol` executable is on your `PATH`.
