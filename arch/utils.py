#!/usr/bin/env python3

import yaml, time
import torch 
import numpy as np
from PIL import Image

def load_config(file):
    with open(file) as f:
        return yaml.full_load(f)
    
def img2tensor(file):
    sample = np.array(Image.open(file).convert('RGB'))
    return torch.tensor(sample.transpose((2, 0, 1))).float()

def chat_playback(response):
    sample = ' '.join(response.split())
    print("Agent: ", end='', flush=True)
    for word in sample:
        time.sleep(0.01)
        print(word, end='', flush=True)
    print()