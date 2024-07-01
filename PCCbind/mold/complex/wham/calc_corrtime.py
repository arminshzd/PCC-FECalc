from multiprocessing import Pool, Manager
from pathlib import Path
import numpy as np
from itertools import repeat
import json

def _get_corr_time(inps):
    traj_id, res, run_dir = inps
    fname = run_dir/f"{traj_id}"/"us.dat"
    with open(fname) as f:
        line = f.readline()
        cnt = []
        while line:
            if line[0] == "#":
                line = f.readline()
                continue
            else:
                line = line.strip().split()
                cnt.append(float(line[1]))
                line = f.readline()
    cnt = np.array(cnt)
    acf = np.array([1]+[np.corrcoef(cnt[:-i], cnt[i:])[0,1] for i in range(1, 20000)])
    try:
        autocorr_time = 2 * (np.where(acf < 1/np.e)[0][0]-1)
    except:
        res[traj_id] = str(np.nan)
        return
    res[traj_id] = str(autocorr_time)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_dir", type=str, help="Directory containing runs")
    parser.add_argument("-n", "--num_runs", type=int, help="Number of runs")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    num_runs = args.num_runs
    with Manager() as manager:
        res = manager.dict()
        with Pool(8) as p:
            p.map(_get_corr_time, zip(list(range(num_runs)), repeat(res, num_runs), repeat(run_dir, num_runs)))
        with open(run_dir/"corr_time.json", "w") as f:
            json.dump(dict(res), f, indent=4)
