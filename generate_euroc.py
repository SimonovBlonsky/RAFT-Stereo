import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from os import path as osp
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

euroc_root = "/data/datasets/EuRoC_mav"
output_root = "/home/chenguyuan/EuRoC_depth"
euroc_scenes = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]
Camera_bf = 47.90639384423901
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # breakpoint()
    img = np.stack([img]*3, axis=-1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()
    scenes = glob.glob(osp.join(euroc_root, '*/'))
    
    for scene in tqdm(sorted(scenes)):
        output_directory = output_root + '/' + scene.split('/')[-2]
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True)
        
        left_imgs = osp.join(scene, 'mav0/cam0/data/*.png')
        right_imgs = osp.join(scene, 'mav0/cam1/data/*.png')
        # breakpoint()

        with torch.no_grad():
            left_images = sorted(glob.glob(left_imgs, recursive=True))
            right_images = sorted(glob.glob(right_imgs, recursive=True))
            print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

            for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)

                _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()
                
                disp = flow_up.cpu().numpy().squeeze()
                depth = Camera_bf / (np.abs(disp) + 1e-6)
                # breakpoint()
                # file_stem = imfile1.split('/')[-2]
                file_stem = imfile1.split('/')[-1]
                file_stem = file_stem.split('.')[0]
                
                # save depth
                if args.save_numpy:
                    np.save(output_directory / f"{file_stem}.npy", depth)
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
