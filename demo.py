import rawpy
import numpy as np
import torch
import argparse
from mwrcanet import Net
from CBUnet import CAUNet, Hist_CAUNet
from config import DENOISING_CHECKPOINT, STAGE_1_CHECKPOINT, STAGE_2_CHECKPOINT
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
from utils import cam_to_ciexyz, rgb_to_gray
from imageio import imwrite

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='input image path', required=True)
    parser.add_argument('-d', '--device', type=int, default=0, help='input image path', required=True)
    opts = parser.parse_args()
    return opts

def get_denoising_model(device):
    model = Net().to(device)
    state_dict = torch.load(DENOISING_CHECKPOINT, map_location='cpu')['state_dict']
    _state_dict = {}
    for k, v in state_dict.items():
        _state_dict[k[len('module.'):]] = v
    model.load_state_dict(_state_dict)
    return model


def get_cbunet(device):
    stage_1_model = CAUNet().to(device)
    state_dict = torch.load(STAGE_1_CHECKPOINT, map_location='cpu')['model_state_dict']
    _state_dict = {}
    for k, v in state_dict.items():
        _state_dict[k[len('module.'):]] = v
    stage_1_model.load_state_dict(_state_dict)   

    stage_2_model = Hist_CAUNet().to(device)
    state_dict = torch.load(STAGE_2_CHECKPOINT, map_location='cpu')['model_state_dict']
    _state_dict = {}
    for k, v in state_dict.items():
        _state_dict[k[len('module.'):]] = v
    stage_2_model.load_state_dict(_state_dict)  
    return stage_1_model, stage_2_model


def read_raw_image(path):
    with rawpy.imread(path) as f:
        black_level = f.black_level_per_channel[0]
        white_level = f.white_level
        ccm = f.color_matrix[:3, :3].astype(np.float32)
        raw = f.raw_image_visible.copy().astype(np.float32)
        bayer_pattern = [_ for _ in f.decode("utf-8")]
        bayer_pattern = bayer_pattern[:2] + bayer_pattern[3] + bayer_pattern[2]
        raw = (raw - black_level) / (white_level - black_level)
        # this lib is slow, may be use numpy if speed is needed
        raw = demosaicing_CFA_Bayer_bilinear(raw, bayer_pattern)
        raw = np.clip(raw, 0, 1)
    return raw, ccm

if __name__ == '__main__':
    args = get_options()
    denoising_model = get_denoising_model(args.device)
    stage_1_model, stage_2_model = get_cbunet(args.device)

    raw, ccm = read_raw_image(args.image)

    with torch.no_grad():
        raw = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).to(args.device)
        raw = denoising_model(raw)

        wb_gain = stage_1_model(raw).mean((2, 3))
        wb_gain = wb_gain[:, 1].unsqueeze(1) / wb_gain

        raw = cam_to_ciexyz(raw, wb_gain, ccm)
        gray_raw = rgb_to_gray(raw)
        gray_rgb = stage_2_model(gray_raw.repeat([1, 4, 11]), gray_raw).mean(1, keepdim=True)
        rgb = raw * gray_rgb / (gray_raw + 1e-6)
        
        rgb = torch.clamp(rgb, 0, 1)
        rgb = rgb.permute([0, 2, 3, 1])
        rgb = rgb.squeeze(0).cpu().numpy()
        rgb = np.uint8(rgb * 255)

    imwrite(args.image[:-4]+'.jpg', rgb)
