from datetime import datetime
from pathlib import Path
import json

from FECalc.TargetMOL import TargetMOL
from FECalc.PCCBuilder import PCCBuilder
from FECalc.FECalc import FECalc
from FECalc.postprocess import postprocess
from FECalc.utils import set_hw_env


with open("system_settings.JSON") as f:
    settings = json.load(f)

# Resolve scheduler and hardware layout (nodes, cores, threads)
scheduler = settings.get("scheduler", "local")
nodes = int(settings.get("nodes", 1))
cores = int(settings.get("cores", 1))
threads = int(settings.get("threads", 1))
nodes, cores, threads = set_hw_env(scheduler, nodes, cores, threads)

PCC_output = Path(settings["PCC_output_dir"])
PCC_settings = Path(settings["PCC_settings_json"])
MOL_settings = Path(settings["MOL_settings_json"])
temperature = float(settings["temperature"])
box_size = float(settings["box_size"])
metad_settings = dict(settings["metad_settings"]) if \
    settings.get("metad_settings", None) is not None else dict()
postprocess_settings = dict(settings["postprocess_settings"]) if \
    settings.get("postprocess_settings", None) is not None else dict()

PCC = PCCBuilder("RWSHR", PCC_output, PCC_settings, nodes=nodes, cores=cores, threads=threads)
PCC.create(check=False)
MOL = TargetMOL(MOL_settings, nodes=nodes, cores=cores, threads=threads)
MOL.create()
complex_output = Path(settings["complex_output_dir"])/f"{PCC.PCC_code}_{MOL.name}"
calculator = FECalc(
    PCC,
    MOL,
    complex_output,
    temperature,
    box=box_size,
    nodes=nodes,
    cores=cores,
    threads=threads,
    **metad_settings,
)
now = datetime.now()
now = now.strftime("%m/%d/%Y, %H:%M:%S")
print(f"Starting {PCC.PCC_code}_{MOL.name} run.")
try:
    calculator.run()
except:
    raise RuntimeError(f"FECalc failed. Check {complex_output} for more info.")

try:
    postprocess(calculator, **postprocess_settings)
finally:
    log_dir = complex_output/"metadata.JSON"
    if log_dir.exists():
        print("INFO: Metadata file exits: OK")
    else:
        raise RuntimeError(f"Somthing went wrong. Check the run directory: {log_dir.parent}")
        
