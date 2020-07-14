import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(v_list):
    return list(map(lambda x: torch.tensor(x).to(DEVICE), v_list))
