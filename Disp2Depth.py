import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm

euroc_disps_path = "/home/chenguyuan/EuRoC_depth/MH_01_easy/*.npy"
output_path = "/home/chenguyuan/EuRoC_depth/MH_01_easy/depth"

Camera_bf = 47.90639384423901

def Disp2Depth(input_dir, output_dir):
    output_directory = Path(output_dir)
    output_directory.mkdir(exist_ok=True)

    input_disps = sorted(glob.glob(input_dir, recursive=True))
    # breakpoint()
    for disps_file in tqdm(list(input_disps)):
        disps = np.load(disps_file)
        depth = Camera_bf / (np.abs(disps) + 1e-6)
        name = disps_file.split('/')[-1].split('.')[0]
        np.save(output_directory / f"{name}.npy", depth)

Disp2Depth(euroc_disps_path, output_path)