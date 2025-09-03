import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson

KBT = 8.314 * 300

def _load_plumed(fname):
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
    if len(data["time"]) < 4000000:
        raise RuntimeError("The simulation might not have completed correctly. Check the md folder.")
    data = pd.DataFrame(data)
    data['weights'] = np.exp(data['pb.bias']*1000/KBT)
    return data

def _get_box_size(gro_fname):
    with open(gro_fname, "r") as f:
        for line in f:
            pass
    return float(line.split()[0])

def _block_anal_3d(x, y, z, weights, block_size=None, folds=None, nbins=100):
    # calculate histogram for all data to get bins
    _, binsout = np.histogramdd([x, y, z], bins=nbins, weights=weights)
    # calculate bin centers
    binsx, binsy, binsz = binsout
    xs = np.round((binsx[1:] + binsx[:-1])/2, 2)
    ys = np.round((binsy[1:] + binsy[:-1])/2, 2)
    zs = np.round((binsz[1:] + binsz[:-1])/2, 2)
    # find block sizes
    if block_size is None:
        if folds is None:
            block_size = 5000*50 #50 ns blocks
            folds = len(x)//block_size
        else:
            block_size = len(x)//folds
    else:
        folds = len(x)//block_size

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
        free_energy = -KBT*np.log(counts)/1000 #kJ/mol
        free_energy_unrolled = []
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    free_energy_unrolled.append(free_energy[i, j, k])
        data[f"f_{fold}"] = free_energy_unrolled
        # Entropy correction along x axis
        data[f"f_{fold}"] += 2*KBT*np.log(data.x)/1000
    
    # de-mean the folds for curve matching
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    for fold in range(folds):
        data[f"f_{fold}"] = data[f"f_{fold}"] - data[f"f_{fold}"].mean()
    
    # calcualte standard deviation and standard error
    data['std'] = data[[f"f_{fold}" for fold in range(folds)]].apply(np.std, axis=1)
    data['ste'] = 1/np.sqrt(folds)*data['std']

    return data

def _calc_region_int(data):
    """
    data = DataFrame with columns x(dcom), y(angle), z(cos), F(free energy)
    """
    data["exp"] = np.exp(-data.F*1000/KBT)
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

    return -KBT*np.log(integrand)/1000
    

def _calc_deltaF(bound_data, unbound_data):
    r_int = _calc_region_int(bound_data.copy())
    p_int = _calc_region_int(unbound_data.copy())
    return r_int - p_int

def _calc_FE(ifdir) -> None:
    colvars1 = _load_plumed(ifdir/"reweight"/"COLVAR") # read colvars
    init_time = 100 # ns
    print(f"INFO: Discarding initial {init_time} ns of data for free energy calculations.")
    colvars1 = colvars1[colvars1['time'] >= init_time * 1000]  # discard the first `init_time` ns of data
    colvars2 = _load_plumed(ifdir/"reweight_extend"/"COLVAR") # read colvars
    colvars = pd.concat([colvars1, colvars2], axis=0)
    # block analysis
    colvars.dropna(inplace=True)
    block_anal_data = _block_anal_3d(colvars.dcom, colvars.ang,
                                        colvars.v3cos, colvars.weights,
                                        nbins=50, block_size=5000*100)
    box_size = _get_box_size(ifdir/"md"/"md.gro")
    unbound_max = box_size/2
    f_list = []
    f_cols = [col for col in block_anal_data.columns if re.match("f_\d+", col)]
    discarded_blocks = 0
    for i in f_cols:
        try:
            # bound = 0<=dcom<=1.5 nm
            bound_data = block_anal_data[(block_anal_data.x>=0.0) & (block_anal_data.x<=1.5)][['x', 'y', 'z', i, 'ste']]
            bound_data.rename(columns={i: 'F'}, inplace=True)
            bound_data.dropna(inplace=True)
            # unbound = 2.0<dcom<box_size/2
            unbound_data = block_anal_data[(block_anal_data.x>2.0) & (block_anal_data.x<unbound_max)][['x', 'y', 'z', i, 'ste']]
            unbound_data.rename(columns={i: 'F'}, inplace=True)
            unbound_data.dropna(inplace=True)
            f_list.append(_calc_deltaF(bound_data=bound_data, unbound_data=unbound_data))
        except (KeyError, ValueError) as e:
            print(f"ERROR: {e}")
            discarded_blocks += 1
            continue
    
    if discarded_blocks != 0:
        print(f"WARNING: {discarded_blocks} block(s) were discarded from the calculations possibly because the system was"\
                " stuck in a bound state for longer than 100 ns consecutively. Check the colvar trajectories.")
    if not f_list:
        raise ValueError("No free energy values could be calculated; f_list is empty.")
    f_list = np.array(f_list)
    return np.nanmean(f_list), np.nanstd(f_list)/np.sqrt(len(f_list)-np.count_nonzero(np.isnan(f_list)))
    
def _calc_K(free_e, free_e_err) -> tuple:
    K = np.exp(-free_e*1000/KBT)
    K_err = K*free_e_err*1000/KBT
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

def postprocess(PCC, target, ifdir) -> None:
    # calc FE
    ifdir = Path(ifdir)
    ifdir = ifdir/f"{PCC}_{target}"/"complex"
    ofname = ifdir.parent/"metadata_extended.JSON"
    if not (ofname).exists():
        free_e, free_e_err = _calc_FE(ifdir)
        # calculate Ks
        K, K_err = _calc_K(free_e, free_e_err)
        # write report
        _write_report(PCC, target, ofname, free_e, free_e_err, K, K_err)
    return None

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PCC postprocess')
    parser.add_argument('PCC', type=str, help="PCC name")
    parser.add_argument('target', type=str, help="target name")
    parser.add_argument('input', type=str, help="input files directory")

    args = parser.parse_args()

    postprocess(args.PCC, args.target, args.input)
