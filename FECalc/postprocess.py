import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson

def _load_plumed(fname, KbT):
    """Load a PLUMED ``COLVAR`` file and compute reweighting factors.

    The file must contain at least the columns ``time``, ``pb.bias``,
    ``dcom``, ``ang`` and ``v3cos``. A ``weights`` column is added based on
    the exponential of ``pb.bias``.

    Args:
        fname (Path or str): Path to the ``COLVAR`` file.
        KbT (float): Thermal energy (``k_B T``) in J/mol.

    Returns:
        pandas.DataFrame: Parsed data including a ``weights`` column.
    """
    data = {}
    with open(fname, 'r') as f:
        fields = f.readline()[:-1].split(" ")[2:]  # skip '#! FIELDS'
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
    data['weights'] = np.exp(data['pb.bias'] * 1000 / KbT)
    return data

def _get_box_size(gro_fname):
    """Return the cubic box edge length from a GROMACS ``.gro`` file.

    Args:
        gro_fname (Path or str): Path to the ``.gro`` structure file.

    Returns:
        float: Box edge length in nanometers.
    """
    with open(gro_fname, "r") as f:
        for line in f:
            pass
    return float(line.split()[0])


def _block_anal_3d(x, y, z, weights, KbT, nbins, folds):
    """Perform block analysis of a 3D free-energy surface.

    The trajectory is divided into ``folds`` blocks, a histogram is computed
    for each block, and the free energy is estimated with reweighting.

    Args:
        x (array-like): dCOM values.
        y (array-like): Angle values.
        z (array-like): Cosine values.
        weights (array-like): Reweighting factors for each frame.
        KbT (float): Thermal energy in J/mol.
        nbins (int or sequence): Number of histogram bins per dimension.
        folds (int): Number of blocks for statistical analysis.

    Returns:
        pandas.DataFrame: Free energies for each block and associated statistics.
    """
    _, binsout = np.histogramdd([x, y, z], bins=nbins, weights=weights)
    binsx, binsy, binsz = binsout
    xs = np.round((binsx[1:] + binsx[:-1]) / 2, 2)
    ys = np.round((binsy[1:] + binsy[:-1]) / 2, 2)
    zs = np.round((binsz[1:] + binsz[:-1]) / 2, 2)
    block_size = len(x) // folds

    data = pd.DataFrame()
    xs_unrolled, ys_unrolled, zs_unrolled = [], [], []
    for i in xs:
        for j in ys:
            for k in zs:
                xs_unrolled.append(i)
                ys_unrolled.append(j)
                zs_unrolled.append(k)

    data['x'] = xs_unrolled
    data['y'] = ys_unrolled
    data['z'] = zs_unrolled

    for fold in range(folds):
        x_fold = x[block_size * fold:(fold + 1) * block_size]
        y_fold = y[block_size * fold:(fold + 1) * block_size]
        z_fold = z[block_size * fold:(fold + 1) * block_size]
        weights_fold = weights[block_size * fold:(fold + 1) * block_size]
        counts, _ = np.histogramdd(
            [x_fold, y_fold, z_fold], bins=[binsx, binsy, binsz], weights=weights_fold)
        counts[counts == 0] = np.nan
        free_energy = -KbT * np.log(counts) / 1000  # kJ/mol
        free_energy_unrolled = []
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    free_energy_unrolled.append(free_energy[i, j, k])
        data[f"f_{fold}"] = free_energy_unrolled
        data[f"f_{fold}"] += 2 * KbT * np.log(data.x) / 1000

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for fold in range(folds):
        data[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"].mean()

    data['std'] = data[[f"f_{fold}" for fold in range(folds)]].apply(np.std, axis=1)
    data['ste'] = 1 / np.sqrt(folds) * data['std']

    return data

def _calc_region_int(data, KbT):
    """Integrate ``exp(-F/kT)`` over a region of the free-energy surface.

    Args:
        data (pandas.DataFrame): Must contain columns ``x`` (dCOM), ``y``
            (angle), ``z`` (cosine), and ``F`` (free energy in kJ/mol).
        KbT (float): Thermal energy in J/mol.

    Returns:
        float: Integral value converted back to free energy (kJ/mol).
    """
    data["exp"] = np.exp(-data.F * 1000 / KbT)
    Z_integrand = {"x": [], "y": [], "exp": []}
    for x in data.x.unique():
        for y in data.y.unique():
            FE_this_xy = data[(data.x == x) & (data.y == y)].copy()
            FE_this_xy.sort_values(by='z', inplace=True)
            Z_integrand["x"].append(x)
            Z_integrand["y"].append(y)
            if FE_this_xy.empty or len(FE_this_xy) == 1:
                Z_integrand["exp"].append(0.0)
            else:
                Z_integrand["exp"].append(
                    simpson(y=FE_this_xy.exp.to_numpy(), x=FE_this_xy.z.to_numpy())
                )

    Z_integrand_pd = pd.DataFrame(Z_integrand)
    Y_integrand = {"x": [], "exp": []}
    for x in Z_integrand_pd.x.unique():
        FE_this_x = Z_integrand_pd[Z_integrand_pd.x == x].copy()
        FE_this_x.sort_values(by='y', inplace=True)
        Y_integrand["x"].append(x)
        if FE_this_x.empty or len(FE_this_x) == 1:
            Y_integrand["exp"].append(0.0)
        else:
            Y_integrand["exp"].append(
                simpson(y=FE_this_x.exp.to_numpy(), x=FE_this_x.y.to_numpy())
            )

    Y_integrand_pd = pd.DataFrame(Y_integrand)
    Y_integrand_pd.sort_values(by='x', inplace=True)
    if len(Y_integrand_pd) <= 1:
        integrand = 1.0
    else:
        integrand = simpson(y=Y_integrand_pd.exp.to_numpy(), x=Y_integrand_pd.x.to_numpy())

    return -KbT * np.log(integrand) / 1000
    

def _calc_deltaF(bound_data, unbound_data, KbT):
    """Compute free-energy difference between bound and unbound regions."""
    r_int = _calc_region_int(bound_data.copy(), KbT)
    p_int = _calc_region_int(unbound_data.copy(), KbT)
    return r_int - p_int

def _calc_FE(ifdir, KbT, init_time, n_folds) -> None:
    """Calculate the binding free energy from reweighted trajectories.

    The routine loads the reweighted ``COLVAR`` file, discards the first
    ``init_time`` nanoseconds, performs block analysis, and integrates the
    bound (dCOM < 2 nm) and unbound regions to obtain the binding free energy
    and its statistical error.

    Args:
        ifdir (Path): Directory containing the simulation output.
        KbT (float): Thermal energy in J/mol.
        init_time (float): Time in nanoseconds to discard from the start of the
            trajectory.
        n_folds (int): Number of blocks for block analysis.

    Returns:
        tuple: Mean free energy (kJ/mol) and its standard error.

    Raises:
        ValueError: If no free-energy values could be calculated.

    Note:
        The bound region is hard-coded as dCOM < 2 nm and the unbound region
        as 2 nm < dCOM < box_size/2.
    """
    colvars = _load_plumed(ifdir/"reweight"/"COLVAR", KbT)
    print(f"INFO: Discarding initial {init_time} ns of data for free energy calculations.")
    colvars = colvars[colvars['time'] >= init_time * 1000]
    colvars.dropna(inplace=True)
    block_anal_data = _block_anal_3d(
        colvars.dcom, colvars.ang, colvars.v3cos, colvars.weights, KbT, 50, n_folds
    )
    box_size = _get_box_size(ifdir/"md"/"md.gro")
    unbound_max = box_size / 2
    f_list = []
    f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
    discarded_blocks = 0
    for i in f_cols:
        try:
            bound_data = block_anal_data[(block_anal_data.x >= 0.0) & (block_anal_data.x <= 1.5)][['x', 'y', 'z', i, 'ste']]
            bound_data.rename(columns={i: 'F'}, inplace=True)
            bound_data.dropna(inplace=True)
            unbound_data = block_anal_data[(block_anal_data.x > 2.0) & (block_anal_data.x < unbound_max)][['x', 'y', 'z', i, 'ste']]
            unbound_data.rename(columns={i: 'F'}, inplace=True)
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
            f"WARNING: {discarded_blocks} block(s) were discarded from the calculations possibly because the system was"
            " stuck in a bound state for longer than 100 ns consecutively. Check the colvar trajectories."
        )
    if not f_list:
        raise ValueError("No free energy values could be calculated; f_list is empty.")
    f_list = np.array(f_list)
    return np.nanmean(f_list), np.nanstd(f_list) / np.sqrt(len(f_list) - np.count_nonzero(np.isnan(f_list)))
    
def _calc_K(free_e, free_e_err, KbT, box_size) -> tuple:
    """Convert free energy to a dissociation constant for a cubic box.

    Args:
        free_e (float): Binding free energy (kJ/mol).
        free_e_err (float): Standard error of the free energy.
        KbT (float): Thermal energy in J/mol.
        box_size (float): Simulation box edge length in nanometers.

    Returns:
        tuple: Equilibrium constant in µM and its uncertainty.
    """
    A = 1 / (6.022e23 * box_size**3 * 1e-24)
    B = 1000 / KbT
    K = A * np.exp(B * free_e) * 1e6  # µM
    K_err = B * K * free_e_err  # µM
    return K, K_err

def _write_report(PCC, target, fname, free_e, free_e_err, K, K_err):
    """Write a JSON summary of the binding free-energy calculation."""
    report = {
        "PCC": PCC,
        "Target": target,
        "FE": free_e,
        "FE_error": free_e_err,
        "K": K,
        "K_err": K_err,
    }
    with open(fname, 'w') as f:
        json.dump(report, f, indent=3)
    return None

def postprocess(fecalc, **kwargs) -> None:
    """Analyse a completed FECalc run and compute binding free energy.

    This is the primary user-facing function. It expects that the PBMetaD
    simulation has been reweighted (see :meth:`FECalc._reweight`) and uses the
    resulting ``COLVAR`` file to compute the binding free energy and
    dissociation constant. Internally, the following steps are performed:

    1. Read the ``COLVAR`` file with :func:`_load_plumed`.
    2. Discard the first ``discard_initial`` nanoseconds of data.
    3. Perform block analysis via :func:`_block_anal_3d` using ``n_folds``
       blocks to estimate statistical uncertainty.
    4. Integrate the bound (dCOM < 2 nm) and unbound regions to obtain the
       free energy (:func:`_calc_FE`).
    5. Convert the free energy to a dissociation constant with
       :func:`_calc_K`.
    6. Write a ``metadata.JSON`` summary with :func:`_write_report`.

    Args:
        fecalc (FECalc): Completed :class:`FECalc` object.
        **kwargs: Optional arguments controlling the analysis.
            discard_initial (float): Time in nanoseconds to discard from the
                start of the trajectory. Defaults to 1 ns.
            n_folds (int): Number of blocks for block analysis. Defaults to 5.

    Returns:
        None
    """
    init_time = float(kwargs.get("discard_initial", 1))  # ns
    n_folds = int(kwargs.get("n_folds", 5))
    KbT = fecalc.KbT
    ifdir = fecalc.complex_dir
    box_size = _get_box_size(ifdir/'md'/'md.gro')
    ofname = ifdir.parent/"metadata.JSON"
    if not ofname.exists():
        free_e, free_e_err = _calc_FE(ifdir, KbT, init_time, n_folds)
        K, K_err = _calc_K(free_e, free_e_err, KbT, box_size)
        _write_report(fecalc.pcc.PCC_code, fecalc.target.name, ofname, free_e, free_e_err, K, K_err)
        return None

def postprocess_wrapper(PCC, target, ifdir, temperature, init_time, n_folds, box_size) -> None:
    """Command-line wrapper around :func:`postprocess`.

    Args:
        PCC (str): PCC identifier.
        target (str): Target molecule name.
        ifdir (str or Path): Input directory containing simulation output.
        temperature (float): Simulation temperature in kelvin.
        init_time (float): Initial time (ns) to discard.
        n_folds (int): Number of blocks for block analysis.
        box_size (float): Simulation box edge length (nm).

    Returns:
        None
    """
    KbT = 8.314 * float(temperature)
    ifdir = Path(ifdir) / "complex"
    ofname = ifdir.parent/"metadata.JSON"
    if not ofname.exists():
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

