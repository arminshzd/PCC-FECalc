import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson


def _load_plumed(fname, KbT):
    """Read a PLUMED ``COLVAR`` file and compute reweighting factors.

    The parser expects, at minimum, the columns ``time``, ``dcom``, ``ang``,
    ``v3cos`` and ``pb.bias``. Additional collective variables are preserved but
    ignored by the subsequent analysis. A ``weights`` column is added to the
    returned table containing the exponential reweighting factor based on the
    metadynamics bias.

    Args:
        fname (Path or str): Path to the ``COLVAR`` file.
        KbT (float): Thermal energy in J/mol used to compute weights.

    Returns:
        pandas.DataFrame: Table containing the collective variables and an
            additional ``weights`` column used for free-energy estimation.
    """
    data = {}
    with open(fname, "r") as f:
        fields = f.readline()[:-1].split(" ")[2:]
        for field in fields:
            data[field] = []
        line = f.readline()
        while line:
            if line[0] == "#":
                line = f.readline()
                continue
            line_list = line.split()
            for i, field in enumerate(fields):
                data[field].append(float(line_list[i]))
            line = f.readline()
    data = pd.DataFrame(data)
    data["weights"] = np.exp(data["pb.bias"] * 1000 / KbT)
    return data

def _get_box_size(gro_fname):
    """Extract the cubic box edge length from a GROMACS ``.gro`` file.

    Args:
        gro_fname (Path or str): Path to the ``.gro`` structure file.

    Returns:
        float: The box edge length in nanometers.
    """
    with open(gro_fname, "r") as f:
        for line in f:
            pass
    return float(line.split()[0])


def _block_anal_3d(x, y, z, weights, KbT, nbins, folds):
    """Perform three-dimensional block analysis of free energies.

    The trajectory is partitioned into ``folds`` contiguous blocks, each used to
    compute an independent estimate of the free-energy surface in ``nbins``
    bins. Standard errors are obtained from the variation across blocks.

    Args:
        x (np.ndarray): Values of the first collective variable.
        y (np.ndarray): Values of the second collective variable.
        z (np.ndarray): Values of the third collective variable.
        weights (np.ndarray): Reweighting factors for each sample.
        KbT (float): Thermal energy in J/mol.
        nbins (int): Number of histogram bins along each dimension.
        folds (int): Number of blocks for the analysis.

    Returns:
        pandas.DataFrame: Free-energy estimates for each bin and block along
            with standard deviations and errors.
    """
    _, binsout = np.histogramdd([x, y, z], bins=nbins, weights=weights)
    binsx, binsy, binsz = binsout
    xs = np.round((binsx[1:] + binsx[:-1]) / 2, 2)
    ys = np.round((binsy[1:] + binsy[:-1]) / 2, 2)
    zs = np.round((binsz[1:] + binsz[:-1]) / 2, 2)
    block_size = len(x) // folds

    data = pd.DataFrame()
    xs_unrolled = []
    ys_unrolled = []
    zs_unrolled = []
    for i in xs:
        for j in ys:
            for k in zs:
                xs_unrolled.append(i)
                ys_unrolled.append(j)
                zs_unrolled.append(k)

    data["x"] = xs_unrolled
    data["y"] = ys_unrolled
    data["z"] = zs_unrolled

    for fold in range(folds):
        x_fold = x[block_size * fold : (fold + 1) * block_size]
        y_fold = y[block_size * fold : (fold + 1) * block_size]
        z_fold = z[block_size * fold : (fold + 1) * block_size]
        weights_fold = weights[block_size * fold : (fold + 1) * block_size]
        counts, _ = np.histogramdd(
            [x_fold, y_fold, z_fold], bins=[binsx, binsy, binsz], weights=weights_fold
        )
        counts[counts == 0] = np.nan
        free_energy = -KbT * np.log(counts) / 1000
        free_energy_unrolled = []
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    free_energy_unrolled.append(free_energy[i, j, k])
        data[f"f_{fold}"] = free_energy_unrolled
        data[f"f_{fold}"] += 2 * KbT * np.log(data.x) / 1000

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for fold in range(folds):
        data[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"] .mean()

    data["std"] = data[[f"f_{fold}" for fold in range(folds)]].apply(np.std, axis=1)
    data["ste"] = 1 / np.sqrt(folds) * data["std"]

    return data

def _calc_region_int(data, KbT):
    """Integrate the Boltzmann weight over a region of CV space.

    Args:
        data (pandas.DataFrame): Data with columns ``x`` (dcom), ``y`` (angle),
            ``z`` (cos) and ``F`` (free energy in kJ/mol).
        KbT (float): Thermal energy in J/mol.

    Returns:
        float: Free energy of the region in kJ/mol.
    """
    data["exp"] = np.exp(-data.F*1000/KbT)
    # integrate over Z
    Z_integrand = {"x": [], "y": [], "exp":[]}
    for x in data.x.unique():
        for y in data.y.unique():
            FE_this_xy = data[(data.x == x) & (data.y == y)].copy()
            FE_this_xy.sort_values(by='z', inplace=True)
            Z_integrand["x"].append(x)
            Z_integrand["y"].append(y)
            if FE_this_xy.empty: # if it doesn't exist, it's empty
                Z_integrand["exp"].append(0.0)
            else:
                Z_integrand["exp"].append(simpson(y=FE_this_xy.exp.to_numpy(), x=FE_this_xy.z.to_numpy()))

    Z_integrand_pd = pd.DataFrame(Z_integrand)
    # integrate over Y
    Y_integrand = {"x": [], "exp":[]}
    for x in Z_integrand_pd.x.unique():
        FE_this_x = Z_integrand_pd[Z_integrand_pd.x == x].copy()
        FE_this_x.sort_values(by='y', inplace=True)
        Y_integrand["x"].append(x)
        if FE_this_x.empty:
            Y_integrand["exp"].append(0.0)
        else:
            Y_integrand["exp"].append(simpson(y=FE_this_x.exp.to_numpy(), x=FE_this_x.y.to_numpy()))

    # integrate over X
    Y_integrand_pd = pd.DataFrame(Y_integrand)
    Y_integrand_pd.sort_values(by='x', inplace=True)
    integrand = simpson(y=Y_integrand_pd.exp.to_numpy(), x=Y_integrand_pd.x.to_numpy())

    return -KbT*np.log(integrand)/1000
    

def _calc_deltaF(bound_data, unbound_data, KbT):
    """Compute the binding free energy difference between two regions."""
    r_int = _calc_region_int(bound_data.copy(), KbT)
    p_int = _calc_region_int(unbound_data.copy(), KbT)
    return r_int - p_int

def _calc_FE(ifdir, KbT, init_time, n_folds) -> None:
    """Estimate the binding free energy from reweighted trajectories.

    The bound state is hard-coded to be configurations where ``dcom`` is less
    than 2 nm, while the unbound reference region is taken from ``2 < dcom <
    2.5`` nm. The routine discards an initial portion of the trajectory,
    performs three-dimensional block analysis and computes the free-energy
    difference between the bound and unbound regions.

    Args:
        ifdir (Path): Directory containing simulation results.
        KbT (float): Thermal energy in J/mol.
        init_time (float): Initial trajectory segment to discard in ns.
        n_folds (int): Number of blocks for statistical analysis.

    Returns:
        tuple(float, float): Mean free energy and its standard error in kJ/mol.
    """
    colvars = _load_plumed(ifdir / "reweight" / "COLVAR", KbT)
    print(
        f"INFO: Discarding initial {init_time} ns of data for free energy calculations."
    )
    colvars = colvars[colvars["time"] >= init_time * 1000]
    colvars.dropna(inplace=True)
    block_anal_data = _block_anal_3d(
        colvars.dcom, colvars.ang, colvars.v3cos, colvars.weights, KbT, 50, n_folds
    )
    f_list = []
    f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
    discarded_blocks = 0
    for i in f_cols:
        try:
            bound_data = block_anal_data[
                (block_anal_data.x >= 0.0) & (block_anal_data.x <= 1.5)
            ][["x", "y", "z", i, "ste"]]
            bound_data.rename(columns={i: "F"}, inplace=True)
            bound_data.dropna(inplace=True)
            unbound_data = block_anal_data[
                (block_anal_data.x > 2.0) & (block_anal_data.x < 2.5)
            ][["x", "y", "z", i, "ste"]]
            unbound_data.rename(columns={i: "F"}, inplace=True)
            unbound_data.dropna(inplace=True)
            f_list.append(
                _calc_deltaF(bound_data=bound_data, unbound_data=unbound_data, KbT=KbT)
            )
        except (KeyError, ValueError) as e:
            print(f"ERROR: {e}")
            discarded_blocks += 1
            continue

    if discarded_blocks != 0:
        print(
            "WARNING: {discarded_blocks} block(s) were discarded from the calculations possibly because the system was"
            " stuck in a bound state for longer than 100 ns consecutively. Check the colvar trajectories.".format(
                discarded_blocks=discarded_blocks
            )
        )
    if not f_list:
        raise ValueError("No free energy values could be calculated; f_list is empty.")
    f_list = np.array(f_list)
    return np.nanmean(f_list), np.nanstd(f_list) / np.sqrt(
        len(f_list) - np.count_nonzero(np.isnan(f_list))
    )
    
def _calc_K(free_e, free_e_err, KbT, box_size) -> tuple:
    """Convert a binding free energy into an equilibrium constant."""
    A = 1 / (6.022e23 * box_size**3 * 1e-24)
    B = 1000 / KbT
    K = A * np.exp(B * free_e) * 1e6
    K_err = B * K * free_e_err
    return K, K_err

def _write_report(PCC, target, fname, free_e, free_e_err, K, K_err):
    """Persist free-energy results to disk in JSON format."""
    report = {
        "PCC": PCC,
        "Target": target,
        "FE": free_e,
        "FE_error": free_e_err,
        "K": K,
        "K_err": K_err,
    }
    with open(fname, "w") as f:
        json.dump(report, f, indent=3)
    return None

def postprocess(fecalc, **kwargs) -> None:
    """Compute binding free energies and equilibrium constants.

    This user-facing routine orchestrates the entire post-processing workflow
    for a ``FECalc`` simulation. It expects that the metadynamics trajectories
    have already been reweighted (via :meth:`FECalc._reweight`) so that a
    ``COLVAR`` file exists in ``<complex>/reweight``. The following operations
    are performed under the hood:

    1. Read optional keyword arguments:
       ``discard_initial`` (float, ns) to specify how much of the initial
       trajectory should be ignored and ``n_folds`` (int) for the number of
       blocks in the statistical analysis. Defaults are 1 ns and 5 blocks.
    2. Obtain simulation temperature as ``KbT`` from ``fecalc`` and determine
       the simulation directory and box size from the GROMACS ``md.gro`` file.
    3. If a ``metadata.JSON`` report does not already exist, call
       :func:`_calc_FE` to estimate the free energy difference between a bound
       region (hard-coded as ``dcom < 2`` nm) and an unbound reference region
       using three-dimensional block analysis.
    4. Convert the free energy to an equilibrium constant with :func:`_calc_K`.
    5. Persist the results—including estimated errors—to ``metadata.JSON`` via
       :func:`_write_report`.

    Args:
        fecalc (FECalc): Finished ``FECalc`` object containing simulation data.
        **kwargs: Optional parameters controlling the analysis. Supported keys
            are ``discard_initial`` (float, ns) and ``n_folds`` (int).

    Returns:
        None
    """
    init_time = float(kwargs.get("discard_initial", 1))
    n_folds = int(kwargs.get("n_folds", 5))
    KbT = fecalc.KbT
    ifdir = fecalc.complex_dir
    box_size = _get_box_size(ifdir / "md" / "md.gro")
    ofname = ifdir.parent / "metadata.JSON"
    if not (ofname).exists():
        free_e, free_e_err = _calc_FE(ifdir, KbT, init_time, n_folds)
        K, K_err = _calc_K(free_e, free_e_err, KbT, box_size)
        _write_report(
            fecalc.pcc.PCC_code, fecalc.target.name, ofname, free_e, free_e_err, K, K_err
        )
    return None

def postprocess_wrapper(PCC, target, ifdir, temperature, init_time, n_folds, box_size) -> None:
    """Convenience wrapper for post-processing outside a ``FECalc`` object."""
    KbT = 8.314 * float(temperature)
    ifdir = Path(ifdir)
    ifdir = ifdir / "complex"
    ofname = ifdir.parent / "metadata.JSON"
    if not (ofname).exists():
        free_e, free_e_err = _calc_FE(ifdir, KbT, float(init_time), int(n_folds))
        K, K_err = _calc_K(free_e, free_e_err, KbT, float(box_size))
        _write_report(PCC, target, ofname, free_e, free_e_err, K, K_err)
    return None

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PCC postprocess')
    parser.add_argument('PCC', type=str, help="PCC name")
    parser.add_argument('target', type=str, help="target name")
    parser.add_argument('input', type=str, help="input files directory")
    parser.add_argument('temperature', type=float, help="simulation temperature")
    parser.add_argument('box_edge', type=float, help="simulation box edge size (nm)")
    parser.add_argument('init_time', type=float, help="simulation length to discard from the begining of the traj")
    parser.add_argument('block_size', type=int, help="block analysis block size")

    args = parser.parse_args()

    postprocess_wrapper(args.PCC, args.target, args.input, args.temperature, args.init_time,
                        args.block_size, args.box_edge)
