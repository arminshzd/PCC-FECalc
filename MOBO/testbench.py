import torch

from MultiObjBayOpt_BoTorch import FreeEnergyEval, MultiObjBayOpt

bounds = [(0, 1), (0, 1)]
eval_model = FreeEnergyEval(bounds=bounds)

