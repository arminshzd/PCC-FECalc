# Example 1

The `example` directory demonstrates how to run a complete binding free energy calculation with PCC-FECalc.  It contains sample settings files and a submission script:

- `pcc_submit_example.py` – orchestrates a full calculation using the library classes.
- `ACT_settings.JSON` and `FEN_settings.JSON` – example target molecule configurations.
- `system_settings.JSON` – describes simulation parameters and paths used by the submission script.
- `ACT.pdb` and `FEN.pdb` – coordinate files for the acetaminophen and fentanyl targets.

To try the example, adjust the paths in `system_settings.JSON` as needed and run:

```bash
python pcc_submit_example.py
```

The script will build the PCC, prepare the target molecule, run the enhanced sampling calculation, and post-process the results to report the binding free energy.
