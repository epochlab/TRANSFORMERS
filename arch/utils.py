#!/usr/bin/env python3

import yaml, torch
import numpy as np
from PIL import Image

def load_config(file):
    with open(file) as f:
        return yaml.full_load(f)
    
def img2tensor(file):
    sample = np.array(Image.open(file).convert('RGB'))
    return torch.tensor(sample.transpose((2, 0, 1))).float()