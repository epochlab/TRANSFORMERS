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

def prune(text):
    soup = ' '.join(text.split())
    sentences = [sentence for sentence in soup.split('. ') if sentence.strip()]

    if sentences and not text.strip().endswith('.'):
        sentences = sentences[:-1]

    pruned_text = '. '.join(sentences)
    if not pruned_text.endswith('.'):
        pruned_text += '.'

    return pruned_text

def chat_playback(response):
    response = response.replace("\n\n", "")
    for word in response:
        time.sleep(0.01)
        print(word, end='', flush=True)
    print()