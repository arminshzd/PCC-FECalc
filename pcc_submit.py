from datetime import datetime
import argparse
from pathlib import Path

from FECalc.FECalc import FECalc

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pcc", help="PCC name")
parser.add_argument("-t", "--target", help="Target molecule")

args = parser.parse_args()

PCC = args.pcc
MOL = args.target
now = datetime.now()
now = now.strftime("%m/%d/%Y, %H:%M:%S")
print(f"Starting {PCC}_{MOL} run.")
calculator = FECalc(PCC, MOL, f"/project2/andrewferguson/armin/FE_DATA/{PCC}_{MOL}", 
                    "/project/andrewferguson/armin/HTVS_Fentanyl/FECalc/FECalc_settings.JSON")
try:
    vals = calculator.run()
finally:
    log_dir = Path(f"/project2/andrewferguson/armin/FE_DATA/{PCC}_{MOL}/metadata.JSON")
    if log_dir.exists():
        print("INFO: Metadata file exits: OK")
    else:
        print("Somthing went wrong. Check the run directory")
        print(f"{log_dir.parent}")
