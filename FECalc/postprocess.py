import re
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from scipy.integrate import simpson

def _load_plumed(fname, KbT):
    data = {}
    with open(fname, 'r') as f:
        fields = f.readline()[:-1].split(" ")[2:] # last char is '/n' and the first two are '#!' and 'FIELDS'
        for field in fields: # add fields to the colvars dict
            data[field] = []
        line = f.readline()
        while line: # read up to LINE_LIM lines
            if line[0] == "#": # Don't read comments
                line = f.readline()
                continue
            line_list = line.split()
            for i, field in enumerate(fields):
                data[field].append(float(line_list[i]))
            line = f.readline()
    #if len(data["time"]) < 4000000:
        #raise RuntimeError("The simulation might not have completed correctly. Check the md folder.")
    data = pd.DataFrame(data)
    data['weights'] = np.exp(data['pb.bias']*1000/KbT)
    return data

def _get_box_size(gro_fname):
    out = subprocess.run(f"tail -n 1 {gro_fname}", shell=True, capture_output=True)
    return float(out.stdout.split()[0])


def _block_anal_3d(x, y, z, weights, KbT, nbins, folds):
    # calculate histogram for all data to get bins
    _, binsout = np.histogramdd([x, y, z], bins=nbins, weights=weights)
    # calculate bin centers
    binsx, binsy, binsz = binsout
    xs = np.round((binsx[1:] + binsx[:-1])/2, 2)
    ys = np.round((binsy[1:] + binsy[:-1])/2, 2)
    zs = np.round((binsz[1:] + binsz[:-1])/2, 2)
    # find block sizes
    block_size = len(x)//folds

    # data frame to store the blocks
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
    
    data['x'] = xs_unrolled
    data['y'] = ys_unrolled
    data['z'] = zs_unrolled

    # calculate free energy for each fold
    for fold in range(folds):
        x_fold = x[block_size*fold:(fold+1)*block_size]
        y_fold = y[block_size*fold:(fold+1)*block_size]
        z_fold = z[block_size*fold:(fold+1)*block_size]
        weights_fold = weights[block_size*fold:(fold+1)*block_size]
        counts, _ = np.histogramdd([x_fold, y_fold, z_fold], bins=[binsx, binsy, binsz], weights=weights_fold)
        counts[counts==0] = np.nan # discard empty bins
        free_energy = -KbT*np.log(counts)/1000 #kJ/mol
        free_energy_unrolled = []
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    free_energy_unrolled.append(free_energy[i, j, k])
        data[f"f_{fold}"] = free_energy_unrolled
        # Entropy correction along x axis
        data[f"f_{fold}"] += 2*KbT*np.log(data.x)/1000
    
    # de-mean the folds for curve matching
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for fold in range(folds):
        data[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"].mean()
    
    # calcualte standard deviation and standard error
    data['std'] = data[[f"f_{fold}" for fold in range(folds)]].apply(np.std, axis=1)
    data['ste'] = 1/np.sqrt(folds)*data['std']

    return data

def _calc_region_int(data, KbT):
    """
    data = DataFrame with columns x(dcom), y(angle), z(cos), F(free energy)
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
    r_int = _calc_region_int(bound_data.copy(), KbT)
    p_int = _calc_region_int(unbound_data.copy(), KbT)
    return r_int - p_int

def _calc_FE(ifdir, KbT, init_time, n_folds) -> None:
    colvars = _load_plumed(ifdir/"reweight"/"COLVAR", KbT) # read colvars
    print(f"INFO: Discarding initial {init_time} ns of data for free energy calculations.")
    init_idx = int(init_time * 10000 // 2)
    colvars = colvars.iloc[init_idx:] # discard the first `init_time` ns of data
    # block analysis
    colvars.dropna(inplace=True)
    block_anal_data = _block_anal_3d(colvars.dcom, colvars.ang, 
                                        colvars.v3cos, colvars.weights, KbT, 
                                        50, n_folds)
    f_list = []
    f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
    discarded_blocks = 0
    for i in f_cols:
        try:
            # bound = 0<=dcom<=1.5 nm
            bound_data = block_anal_data[(block_anal_data.x>=0.0) & (block_anal_data.x<=1.5)][['x', 'y', 'z', i, 'ste']]
            bound_data.rename(columns={i: 'F'}, inplace=True)
            bound_data.dropna(inplace=True)
            # unbound = 2.0<dcom<2.4~inf nm 
            unbound_data = block_anal_data[(block_anal_data.x>2.0) & (block_anal_data.x<2.5)][['x', 'y', 'z', i, 'ste']]
            unbound_data.rename(columns={i: 'F'}, inplace=True)
            unbound_data.dropna(inplace=True)
            f_list.append(_calc_deltaF(bound_data=bound_data, unbound_data=unbound_data, KbT=KbT))
        except:
            discarded_blocks += 1
            continue
    
    if discarded_blocks != 0:
        print(f"WARNING: {discarded_blocks} block(s) were discarded from the calculations possibly because the system was"\
                " stuck in a bound state for longer than 100 ns consecutively. Check the colvar trajectories.")
    f_list = np.array(f_list)
    return np.nanmean(f_list), np.nanstd(f_list)/np.sqrt(len(f_list)-np.count_nonzero(np.isnan(f_list)))
    
def _calc_K(free_e, free_e_err, KbT, box_size) -> tuple:
    # CUBIC BOX ONLY
    A = 1/(6.022e23*box_size**3*1e-24)# C0 (M) for box edge = `box_size`
    B = 1000/KbT
    K = A*np.exp(B*free_e)*1e6 # muM
    K_err = B*K*free_e_err # muM
    return K, K_err

def _write_report(PCC, target, fname, free_e, free_e_err, K, K_err):
    report = {
        "PCC": PCC,
        "Target": target,
        "FE": free_e,
        "FE_error": free_e_err,
        "K": K,
        "K_err": K_err
    }
    with open(fname, 'w') as f:
        json.dump(report, f, indent=3)
    return None

def postprocess(fecalc, **kwargs) -> None:
    init_time = float(kwargs.get("discard_initial", 1)) # in ns
    n_folds = int(kwargs.get("n_folds", 5)) # # folds
    KbT = fecalc.KbT
    ifdir = fecalc.complex_dir
    box_size = _get_box_size(ifdir/'md'/'md.gro')
    ofname = ifdir.parent/"metadata.JSON"
    if not (ofname).exists():
        free_e, free_e_err = _calc_FE(ifdir, KbT, init_time, n_folds)
        # calculate Ks
        K, K_err = _calc_K(free_e, free_e_err, KbT, box_size)
        # write report
        _write_report(fecalc.pcc.PCC_code, fecalc.target.name, ofname, free_e, free_e_err, K, K_err)
        return None

def postprocess_wrapper(PCC, target, ifdir, temperature, init_time, n_folds, box_size) -> None:
    KbT = 8.314 * float(temperature)
    # calc FE
    ifdir = Path(ifdir)
    ifdir = ifdir/"complex"
    ofname = ifdir.parent/"metadata.JSON"
    if not (ofname).exists():
        free_e, free_e_err = _calc_FE(ifdir, KbT, float(init_time), int(n_folds))
        # calculate Ks
        K, K_err = _calc_K(free_e, free_e_err, KbT, float(box_size))
        # write report
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
