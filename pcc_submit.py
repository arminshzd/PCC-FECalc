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
output_path = Path("/scratch/midway2/arminsh/FE_DATA/")
#output_path = Path("/project2/andrewferguson/armin/FE_DATA/")
print(f"Starting {PCC}_{MOL} run.")
calculator = FECalc(PCC, MOL, str(output_path/f"{PCC}_{MOL}"), 
                    "./FECalc/FECalc_settings.JSON")
try:
    vals = calculator.run()
finally:
    log_dir = output_path/f"{PCC}_{MOL}"/"metadata.JSON"
    if log_dir.exists():
        print("INFO: Metadata file exits: OK")
    else:
        print("Somthing went wrong. Check the run directory")
        print(f"{log_dir.parent}")
