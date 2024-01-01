#!/usr/bin/env python3

from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_PATH = "/mnt/artemis/library/weights/blip/blip-image-captioning-large"

class Blip():
    def __init__(self, device):
        self.processor = BlipProcessor.from_pretrained(MODEL_PATH)
        self.model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
        self.device = device

    def witness(self, img, max_length=128):
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.processor.decode(out[0], skip_special_tokens=True)