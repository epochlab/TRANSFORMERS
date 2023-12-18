#!/usr/bin/env python3

import yaml

def load_config(file):
    with open(file) as f:
        return yaml.full_load(f)