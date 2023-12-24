#!/usr/bin/env python3

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils import url2image, albedo

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/Users/James/Documents/blip-image-captioning-large/"

class Vision():
    def __init__(self, model_dir, device):
        self.processor = BlipProcessor.from_pretrained(model_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
        self.device = device

    def witness(self, img, max_length=128):
        inputs = self.processor(img, return_tensors="pt").to(DEVICE)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.processor.decode(out[0], skip_special_tokens=True)