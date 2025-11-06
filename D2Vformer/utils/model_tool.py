import torch
import numpy as np
def get_item(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().item()
    else:
        return data