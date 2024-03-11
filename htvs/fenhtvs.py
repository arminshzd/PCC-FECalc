import os
import re
import json
from pathlib import Path
from multiprocessing import Pool
import shutil
import pickle
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf_discrete

from ..FECalc.FECalc import FECalc
from ..MFMOBO.gskgpr import GaussianStringKernelGP
from ..MFMOBO.seq2ascii import Seq2Ascii


class FenHTVS:
    """
    Class for high throughput screening of PCCs for fentanyl detection. Given a set of initial calculations, finds the
    best new candidates from the full chemical space of 5-AA long PCCs, calculates binding free energies automatically
    (FECalc), and retrains the model to find new optimal candidates (MFMOBO).
    """

    def __init__(self, settings_dir: str):
        # define the reference point
        self.settings_dir = settings_dir
        with open(self.settings_dir) as f:
            self.settings = json.load(f)
        self.ref_point = torch.Tensor([-50, -50])
        # create a translator
        self.translator = Seq2Ascii(self.settings["blosum_dir"])
        # fit the translator with the full chemical space
        self._load_chem_space(self.settings["full_space_dir"])
        # set choices to all PCCs in the chemical space
        self.choices = torch.as_tensor(list(self.translator.int2str.keys()))
        # defining necessary pieces of the MFMOBO pipeline
        self.optimizer = None
        self.mll = None
        self.model = None
        self.database = None
        self.train_x = None
        self.train_y = None
        self.err_y = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _read_json(pcc, data_dir: str):
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

    def _load_data(self, data_dir: str):
        pcc_list = []
        for folder in os.listdir(data_dir):
            if re.match("[A-Z]{5}_[A-Z]{3}", folder):
                pcc_list.append(folder.split("_")[0])

        pcc_list = set(pcc_list)
        data = []
        for pcc in pcc_list:
            try:
                data.append(pd.DataFrame(self._read_json(pcc, data_dir)))
            except Exception as e:
                print(f"{e}: Incomplete data for {pcc}.")

        data = pd.concat(data)
        data.reset_index(inplace=True, drop=True)
        return data

    def _init_model(self):
        if not (self.train_x and self.train_y and self.err_y):
            raise RuntimeError("Model is not initialized. Call model.init_pipeline if you are fitting with initial data"
                               "or model.load_model if continuing a previous run.")
        models = [
            GaussianStringKernelGP(train_x=self.train_x, train_y=self.train_y[:, 0],
                                   likelihood=FixedNoiseGaussianLikelihood(noise=self.err_y[:, 0]),
                                   translator=self.translator),
            GaussianStringKernelGP(train_x=self.train_x, train_y=self.train_y[:, 1],
                                   likelihood=FixedNoiseGaussianLikelihood(noise=self.err_y[:, 1]),
                                   translator=self.translator)
        ]
        self.model = ModelListGP(*models).to(self.device)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model).to(self.device)
        return None

    def init_pipeline(self, data_dir: str):
        self.database = self._load_data(data_dir)
        self.database["ddG_sen"] = -1 * self.database.F_FEN
        self.database.ddG_sen = (self.database.ddG_sen - self.database.ddG_sen.mean()) / self.database.ddG_sen.std()
        self.database["sen_var"] = self.database.err_FEN
        self.database.sen_var = self.database.sen_var / self.database.ddG_sen.std()
        self.database["ddG_spe"] = self.database.F_DEC - self.database.F_FEN
        self.database.ddG_spe = (self.database.ddG_spe - self.database.ddG_spe.mean()) / self.database.ddG_spe.std()
        self.database["spe_var"] = np.sqrt(self.database.err_FEN ** 2 + self.database.err_DEC ** 2)
        self.database.spe_var = self.database.spe_var / self.database.ddG_spe.std()

        self.train_x = self.translator.encode_to_int(self.database.PCC.to_list()).to(self.device)
        fe_sen = torch.tensor(self.database.ddG_sen.to_numpy()).float().to(self.device)
        fe_sen_var = torch.tensor(self.database.sen_var.to_numpy()).float().to(self.device)
        fe_spe = torch.tensor(self.database.ddG_spe.to_numpy()).float().to(self.device)
        fe_spe_var = torch.tensor(self.database.spe_var.to_numpy()).float().to(self.device)
        self.train_y = torch.cat([fe_sen.view(-1, 1), fe_spe.view(-1, 1)], dim=1)
        self.err_y = torch.cat([fe_sen_var.view(-1, 1), fe_spe_var.view(-1, 1)], dim=1)
        self.choices = self.choices[~torch.isin(self.choices, self.train_x)]

    def _load_chem_space(self, full_space_dir: str):
        """
        load the full chemical space and fit the translator.

        Args:
            full_space_dir (str): path to the text file containing all 5-AA PCCs

        Returns:
            None
        """
        full_space = []
        with open(full_space_dir) as f:
            line = f.readline()
            while line:
                full_space.append(line.split()[0])
                line = f.readline()

        self.translator.fit(full_space)
        return None

    def _fit_gpytorch_model(self, n_iters: int = 100):
        for i in range(n_iters):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.mll.model(*self.model.train_inputs)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.mll.model.train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   sigma11: %.3f   sigma21: %.3f   sigma12: %.3f   sigma22: %.3f' % (
                i + 1, n_iters, loss.item(),
                self.mll.model.models[0].covar_module.sigma1.item(),
                self.mll.model.models[0].covar_module.sigma2.item(),
                self.mll.model.models[1].covar_module.sigma1.item(),
                self.mll.model.models[1].covar_module.sigma2.item(),
            ))
            self.optimizer.step()
        return None

    def opt_qehvi_get_obs(self, sampler: SobolQMCNormalSampler, q: int = 3):
        with torch.no_grad():
            pred = self.model.posterior(self.train_x).mean

        partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point,
            Y=pred
        )

        acq_func = qExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )

        # optimize
        candidates, _ = optimize_acqf_discrete(
            acq_function=acq_func,
            q=q,
            choices=self.choices,
            max_batch_size=100,
            unique=True
        )
        return candidates.detach()

    def _find_new_candidates(self, q: int = 3) -> list:
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        #self._fit_gpytorch_model()
        fit_gpytorch_mll(self.mll)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        outputs = []
        new_x = self.opt_qehvi_get_obs(sampler=sampler, q=q)
        return self.translator.decode(new_x.squeeze())

    def _sub_fe_calc(self, pcc: str, calc_dir: str, log_dir: str) -> list:
        log_dir = Path(log_dir)

        def _sub_pcc_mol(mol) -> list[Union[str, list]]:
            calculator = FECalc(pcc, mol, Path(f"{calc_dir}/{pcc}_{mol}"), Path(self.settings_dir))
            try:
                vals = calculator.run()
                return [pcc, mol, *vals]  # PCC name, target name, FE, FE error, K, K_err
            except Exception as e:
                print(f"{e}: Free energy calculation for {pcc} and {mol} failed. Check the log file: {log_dir}")
            finally:
                if log_dir.exists():
                    print("INFO: Metadata file exits: OK")
                else:
                    raise RuntimeError(f"Somthing went wrong. Check the run directory: {log_dir.parent}")

        with Pool(2) as p:
            res = p.map(_sub_pcc_mol, ['FEN', 'DEC'])
        return res

    def _update_database(self, calc_dir: str, candidates: list) -> None:
        # copy the metadata files to the data directory
        calc_dir = Path(calc_dir)
        data_dir = self.settings['data_dir']
        for pcc in candidates:
            for target in ["FEN", "DEC"]:
                shutil.copy(calc_dir/f"{pcc}_{target}"/"metadata.JSON", data_dir/f"{pcc}_{target}.JSON")
        # re-import the database (cannot simply append because the normalization of the data changes as new
        # data is added.
        self.database = self._load_data(data_dir)

    def _save(self, file_name: str) -> None:
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    def _make_plot(self, new_x, save_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        post_pred = self.model.posterior(self.choices)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(x = post_pred.mean[:, 0].detach().numpy(), y = post_pred.mean[:, 1].detach().numpy(), color='blue', label="Predicted")
        ax.scatter(x = self.train_y[:, 0].detach().numpy(), y = self.train_y[:, 1].detach().numpy(), color='red', label="Training")
        new_x = self.translator.encode_to_int(new_x).to(self.device)
        post_chosen = self.model.posterior(new_x)
        ax.scatter(x = post_chosen.mean[new_x, 0].detach().numpy(), y = post_chosen.mean[new_x, 1].detach().numpy(), color='green', label="New Candidates")
        ax.xlabel(r"-$\Delta G_{FEN}$")
        ax.ylabel(r"$\Delta G_{DEC}-\Delta G_{FEN}$")
        plt.legend()
        plt.savefig(save_dir)


    def step(self, calc_dir: str, log_dir: str, save_dir: Union[str, Path]) -> list:
        # initiate the model
        self._init_model()
        # fit the GPR with current data and find new candidates
        new_x = self._find_new_candidates(q=self.settings['q'])
        print("New candidates for next round: ")
        for i in new_x:
            print(f"{i} ", end="")
        print("")
        # submit free energy calculations
        print("Submitting new free energy calculations")
        with Pool(3) as p:
            _ = p.map(lambda pcc: self._sub_fe_calc(pcc, calc_dir=calc_dir, log_dir=log_dir), new_x)
        # update database
        self._update_database(calc_dir, new_x)
        # save the current model
        self._save(save_dir)
        return new_x

    def run(self, n_steps: int, calc_dir: str, log_dir: str, save_dir: str) -> None:
        # TODO: incomplete
        for i in range(n_steps):
            new_x = self.step(calc_dir=calc_dir, log_dir=log_dir, save_dir=Path(save_dir)/f"step_{i}.pkl")
            self._make_plot(new_x=new_x, save_dir=Path(save_dir)/f"step_{i}.png")

