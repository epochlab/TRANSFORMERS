#!/usr/bin/env python3

import yaml, time, requests
from io import StringIO
from contextlib import redirect_stdout

import torch
import numpy as np
from PIL import Image

def device_mapper():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def load_config(file: str):
    with open(file) as f:
        return yaml.full_load(f)

def url2image(url: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return Image.open(response.raw).convert('RGB')
    else:
        print(f"Error: URL returned status code {response.status_code}")

def image2tensor(file: str):
    sample = np.array(Image.open(file).convert('RGB'))
    # To Do: Normalise
    return torch.tensor(sample.transpose((2, 0, 1))).float()

def albedo(text, clr='white'):
    color_index = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(clr.lower())
    base_code = 30 + color_index
    color_code = f"\u001b[{base_code}m"
    return f"{color_code}{text}\u001b[0m"

def chat_playback(response):
    bucket = ' '.join(response.split())
    for word in bucket:
        time.sleep(0.01)
        print(albedo(f"{word}", "cyan"), end='', flush=True)
    print("\n")

def coder(response):
    # USER WARNING: Consider AI safety before execution.
    if '```' in response:
        python_code = response.split('```\n')[1].split("```")[0]
        print(albedo(f"CODE DETECTED:\n\n{python_code}", 'red'))

        if input("Do you want to EXECUTE? (Y/n) ").lower() == 'y':
            print("RUNNING...")
            term = StringIO()
            with redirect_stdout(term): exec(python_code)
            print(term.getvalue().strip())