from datetime import datetime
import argparse
from pathlib import Path

from PCCFEcalc import PCCFEcalc

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pcc1", help="PCC1 name")
parser.add_argument("-t", "--pcc2", help="PCC2 name")

args = parser.parse_args()

PCC1 = args.pcc1
PCC2 = args.pcc2
now = datetime.now()
now = now.strftime("%m/%d/%Y, %H:%M:%S")
print(f"Starting {PCC1}_{PCC2} run.")
output_path = Path("/scratch/midway2/arminsh/PCC_PCC_DATA")
calculator = PCCFEcalc(PCC1, PCC2, str(output_path/f"{PCC1}_{PCC2}"),
                    "/project/andrewferguson/armin/HTVS_Fentanyl/PCCbind/PCCPCC_settings.JSON")
try:
    vals = calculator.run()
finally:
    log_dir = output_path/f"{PCC1}_{PCC2}"/"metadata.JSON"
    if log_dir.exists():
        print("INFO: Metadata file exits: OK")
    else:
        print("Somthing went wrong. Check the run directory")
        print(f"{log_dir.parent}")
