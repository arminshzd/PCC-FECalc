from datetime import datetime
import argparse
from pathlib import Path

from PCCBuilder import PCCBuilder

# Read User Input
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--pcc", help="PCC name")

args = parser.parse_args()

PCC = args.pcc
now = datetime.now()
now = now.strftime("%m/%d/%Y, %H:%M:%S")
print(f"Starting PCCBuilder for {PCC} run.")
output_path = Path("/scratch/midway2/arminsh/PCC_DB")
builder = PCCBuilder(PCC, str(output_path/f"{PCC}"),
                    "/project/andrewferguson/armin/HTVS_Fentanyl/PCCbind/PCCBuilder_settings.JSON")
builder.create()