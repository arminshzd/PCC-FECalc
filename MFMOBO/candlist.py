# %%
import json
import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
import botorch
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement, qExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
import gpytorch
from tqdm import tqdm

from gskgpr import GaussianStringKernelGP
from seq2ascii import Seq2Ascii

# %%
REF_POINT = torch.Tensor([-50, -50])
gpytorch.settings.debug._set_state(True)
botorch.settings.debug._set_state(True)

# %%
def load_json_res(pcc, data_dir):
    with open(f"{data_dir}/{pcc}_FEN.JSON") as f:
        rep = json.load(f)
    F_fen = rep["FE"]
    F_fen_err = rep["FE_error"]

    with open(f"{data_dir}/{pcc}_DEC.JSON") as f:
        rep = json.load(f)
    F_dec = rep["FE"]
    F_dec_err = rep["FE_error"]
    return {"PCC": [rep["PCC"]], "F_FEN": [float(F_fen)], "err_FEN": [float(F_fen_err)],
             "F_DEC": [float(F_dec)], "err_DEC": [float(F_dec_err)]}

def load_data(data_dir):
    PCC_list = []
    for folder in os.listdir(data_dir):
        if re.match("[A-Z]{5}_[A-Z]{3}", folder):
            PCC_list.append(folder.split("_")[0])

    PCC_list = set(PCC_list)
    data = []
    for pcc in PCC_list:
        try:
            data.append(pd.DataFrame(load_json_res(pcc, data_dir)))
        except:
            print(f"Skipping {pcc}.")

    data = pd.concat(data)
    data.reset_index(inplace=True, drop=True)
    return data

# %%
dataset = load_data("/Users/arminsh/Documents/FEN-HTVS/results")
dataset["ddG_sen"] = -1*dataset.F_FEN
dataset.ddG_sen = (dataset.ddG_sen - dataset.ddG_sen.mean())/dataset.ddG_sen.std()
dataset["sen_var"] = dataset.err_FEN
dataset.sen_var = dataset.sen_var/dataset.ddG_sen.std()
dataset["ddG_spe"] = dataset.F_DEC-dataset.F_FEN
dataset.ddG_spe = (dataset.ddG_spe - dataset.ddG_spe.mean())/dataset.ddG_spe.std()
dataset["spe_var"] = np.sqrt(dataset.err_FEN**2 + dataset.err_DEC**2)
dataset.spe_var = dataset.spe_var/dataset.ddG_spe.std()
# %%
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = Seq2Ascii("/Users/arminsh/Documents/FEN-HTVS/MFMOBO/AA.blosum62.pckl")

fspace = []
with open("/Users/arminsh/Documents/FEN-HTVS/gen_input_space/full_space.txt") as f:
    line = f.readline()
    while line:
        fspace.append(line.split()[0])
        line = f.readline()

translator.fit(fspace)
# %%
encoded_x = translator.encode_to_int(dataset.PCC.to_list()).to(device)
FE_sen = torch.tensor(dataset.ddG_sen.to_numpy()).float().to(device)
FE_sen_var = torch.tensor(dataset.sen_var.to_numpy()).float().to(device)
FE_spe = torch.tensor(dataset.ddG_spe.to_numpy()).float().to(device)
FE_spe_var = torch.tensor(dataset.spe_var.to_numpy()).float().to(device)
train_y = torch.cat([FE_sen.view(-1, 1), FE_spe.view(-1, 1)], dim=1)
err_y = torch.cat([FE_sen_var.view(-1, 1), FE_spe_var.view(-1, 1)], dim=1)
# %%
def initialize_model(train_x, train_y, err_y, translator):
    models = [
        GaussianStringKernelGP(train_x=train_x, train_y=train_y[:, 0], 
                            likelihood=FixedNoiseGaussianLikelihood(noise=err_y[:, 0]), 
                            translator=translator),
        GaussianStringKernelGP(train_x=train_x, train_y=train_y[:, 1],
                            likelihood=FixedNoiseGaussianLikelihood(noise=err_y[:, 1]), 
                            translator=translator)
    ]
    model = ModelListGP(*models).to(device)
    mll = SumMarginalLogLikelihood(model.likelihood, model).to(device)
    return model, mll
# %%
def fit_gpytorch_model(mll, optimizer, n_iters=100):
    for i in range(n_iters):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = mll.model(*model.train_inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, mll.model.train_targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   sigma11: %.3f   sigma21: %.3f   sigma12: %.3f   sigma22: %.3f' % (
            i + 1, n_iters, loss.item(),
            mll.model.models[0].covar_module.sigma1.item(),
            mll.model.models[0].covar_module.sigma2.item(),
            mll.model.models[1].covar_module.sigma1.item(),
            mll.model.models[1].covar_module.sigma2.item(),
        ))
        optimizer.step()
# %%
def opt_qehvi_get_obs(model, train_x, choices, sampler):
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    
    partitioning = FastNondominatedPartitioning(
        ref_point=REF_POINT,
        Y=pred
    )

    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=REF_POINT,
        partitioning=partitioning,
        sampler=sampler,
    )

    # acq_func = qNoisyExpectedHypervolumeImprovement(
    #     model=model,
    #     ref_point=REF_POINT,
    #     X_baseline=train_x,
    #     sampler=sampler,
    # )

    # optimize
    candidates, _ = optimize_acqf_discrete(
        acq_function=acq_func,
        q=3,
        choices=choices,
        max_batch_size=100,
        unique=True
    )
    # observe new values
    new_x = candidates.detach()
    new_obj_true = None #get the true value
    new_obj_true_err = None #get the true value error
    new_post = model.posterior(new_x) 
    new_obj = new_post.mean
    new_obj_err = new_post.variance
    return new_x, new_obj_true, new_obj_true_err, new_obj, new_obj_err


# %%
model, mll = initialize_model(encoded_x, train_y, err_y, translator)
post = model.posterior(encoded_x)
obj_vals = post.mean
obj_errs = post.variance

hvs = []

bd = DominatedPartitioning(ref_point=REF_POINT, Y=train_y)
volume = bd.compute_hypervolume().item()
hvs.append(volume)

choices = list(translator.int2str.keys())
for i in dataset.PCC: # remove the ones that are already in the training set
    choices.remove(translator.str2int[i])
choices = torch.Tensor(choices).view(-1, 1).to(device)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for iteration in range(0, 1):
    #fit_gpytorch_mll(mll)
    fit_gpytorch_model(mll, optimizer, n_iters=100)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
    outputs = []
    new_x, new_obj_true, new_obj_true_err, new_obj, new_obj_err = opt_qehvi_get_obs(model, encoded_x, choices, sampler)
    # update training data
    #train_x = torch.cat([train_x, new_x], dim=0)
    #obj_vals = torch.cat([obj_vals, new_obj], dim=0)
    #obj_errs = torch.cat([obj_errs, new_obj_err], dim=0)
    #train_y = torch.cat([train_y, new_obj_true], dim=0)

    # recompute hvs
    #bd = DominatedPartitioning(ref_point=REF_POINT, Y=train_y)
    #volume = bd.compute_hypervolume().item()
    #hvs.append(volume)
    print(new_x)
    print(translator.decode(new_x.squeeze()))