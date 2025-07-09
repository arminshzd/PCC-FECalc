from datetime import datetime
import argparse
from pathlib import Path
import json

from FECalc.TargetMOL import TargetMOL
from FECalc.PCCBuilder import PCCBuilder
from FECalc.FECalc import FECalc
from FECalc.postprocess import postprocess


# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pcc", help="PCC name")
parser.add_argument("-s", "--settings", help="Settings file")
args = parser.parse_args()

with open(args.settings) as f:
    settings = json.load(f)

PCC_output = Path(settings["PCC_output_dir"])
PCC_settings = Path(settings["PCC_settings_json"])
MOL_settings = Path(settings["MOL_settings_json"])
temperature = Path(settings["temperature"])

PCC = PCCBuilder(args.pcc, PCC_output, PCC_settings)
PCC.create()
MOL = TargetMOL(MOL_settings)
MOL.create()
complex_output = Path(settings["complex_output_dir"])/f"{PCC.PCC_code}_{MOL.name}"
calculator = FECalc(PCC, MOL, complex_output, temperature)
now = datetime.now()
now = now.strftime("%m/%d/%Y, %H:%M:%S")
print(f"Starting {PCC.PCC_code}_{MOL.name} run.")
try:
    calculator.run()
except:
    print(f"FECalc failed. Check {complex_output} for more info.")

try:
    postprocess(calculator)
finally:
    log_dir = complex_output/"metadata.JSON"
    if log_dir.exists():
        print("INFO: Metadata file exits: OK")
    else:
        print("Somthing went wrong. Check the run directory")
        print(f"{log_dir.parent}")
