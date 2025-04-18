from tqdm.auto import tqdm

AAs = ['D', 'S', 'K', 'P', 'T', 'F', 'N', 'G', 'H', 'L', 'R', 'W', 'A', 'V', 'E', 'Y']
full_list = []
for a0 in tqdm(AAs):
    with open(f"./{a0}.txt") as f:
        for line in f:
            full_list.append(line)
    print(f"Done with {a0}.")

pbar = tqdm(desc="Writing", total=16**5)
cntr = 0
with open(f"./full_space.txt", 'w') as f:
    for line in full_list:
        if line.strip():
            f.write(line)
        cntr += 1
        pbar.update(cntr)
pbar.close()

