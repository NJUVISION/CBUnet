from numpy import isin
import torch

def cam_to_ciexyz(raw, wb, ccm):
    assert isinstance(raw, torch.Tensor)
    assert len(wb.shape) == 2 and len(ccm.shape) == 2
    raw = raw * wb.unsqueeze(2).unsqueeze(2)
    raw = torch.clamp(raw, 0, 1)
    b, _, h, w = raw.shape
    raw = raw.permute([0, 2, 3, 1])
    ciexyz = torch.matmul(raw.reshape([-1, 3]), ccm.T).reshape([b, h, w, 3])
    ciexyz = ciexyz.permute([0, 3, 1, 2])
    return ciexyz

def rgb_to_gray(rgb):
    assert isinstance(rgb, torch.Tensor) and len(rgb.shape) == 4
    gray = rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
    gray = gray.unsqueeze(1)
    return gray
