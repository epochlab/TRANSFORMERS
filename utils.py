#!/usr/bin/env python3

import yaml, time
from io import StringIO
from contextlib import redirect_stdout

import torch

def device_mapper():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def load_config(file: str):
    with open(file) as f:
        return yaml.full_load(f)

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