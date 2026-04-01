import torch

def concat_input(z_or_x, y):
    if len(y.shape) == 1:
        y = y.unsqueeze(1)
    return torch.cat([z_or_x, y.float()], dim=1)
