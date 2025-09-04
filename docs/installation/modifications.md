# Modifications

The default installation covers the core features of PCC-FECalc. Depending on your workflow, you may wish to install optional extras or adjust configuration files:

- **Documentation build**: `pip install .[docs]` installs `mkdocs` and related plugins for building the documentation.
- **Testing**: `pip install .[test]` installs `pytest` for running the test suite.
- **Scheduler settings**: edit the `system_settings.JSON` example to match your cluster's scheduler (e.g., `slurm`, `pbs`, or `lsf`).

These adjustments allow the package to integrate with different environments and use cases.
