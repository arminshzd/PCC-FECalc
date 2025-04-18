from multiprocessing import Pool
from tqdm import tqdm

AAs = ['D', 'S', 'K', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y']
Xs = []

def create_file(a0):
    pbar = tqdm(desc="Progress:", total = 16**4)
    with open(f"./{a0}.txt", 'w') as f:
        cntr = 0
        for a1 in AAs:
            for a2 in AAs:
                for a3 in AAs:
                    for a4 in AAs:
                        x = "".join([a0, a1, a2, a3, a4])
                        if x not in Xs:
                            Xs.append(x)
                            f.write(x+"\n")
                            cntr += 1
                            pbar.update(cntr)

if __name__ == '__main__':
    with Pool(16) as p:
        p.map(create_file, AAs)