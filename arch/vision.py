#!/usr/bin/env python3

from transformers import BlipProcessor, BlipForConditionalGeneration

class Vision():
    def __init__(self, model_dir, device):
        self.processor = BlipProcessor.from_pretrained(model_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
        self.device = device

    def witness(self, img, max_length=128):
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.processor.decode(out[0], skip_special_tokens=True)