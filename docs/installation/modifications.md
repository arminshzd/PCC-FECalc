# Modifications

AmberTools' `sqm` executable is required for parameter generation and depends on `libgfortran5`. Install them after setting up the repository:

```bash
conda install -c conda-forge ambertools libgfortran5
```

PyMOL is required but not installed automatically because no pip wheel is available. Install PyMOL separately from [pymol.org](https://www.pymol.org/) or build from the open-source [GitHub repository](https://github.com/schrodinger/pymol-open-source). There's also a conda package package available:

```bash
conda install -c conda-forge pymol-open-source
```

Ensure the `pymol` executable is on your `PATH`.

The default installation covers the core features of PCC-FECalc. Depending on your workflow, you may wish to install optional extras or adjust configuration files:

- **Documentation build**: `pip install .[docs]` installs `mkdocs` and related plugins for building the documentation.
- **Testing**: `pip install .[test]` installs `pytest` for running the test suite.
- **Scheduler settings**: edit the `system_settings.JSON` example to match your cluster's scheduler (e.g., `slurm`, `pbs`, or `lsf`).

These adjustments allow the package to integrate with different environments and use cases.
