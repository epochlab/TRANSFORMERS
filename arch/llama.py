#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, pipeline

class Llama():
    def __init__(self):
        self.model = "/mnt/artemis/library/weights/meta/llama-2/7Bf/hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )
        return sequences

    def prune(self, seq):
        if not seq:
            raise ValueError("Error: Empty sequence generated.")

        text = seq[0]['generated_text'].replace('\n', ' ')
        sentences = [sentence for sentence in text.split('. ') if sentence.strip()][1:]

        if sentences and not text.strip().endswith('.'):
            sentences = sentences[:-1]

        pruned_text = '. '.join(sentences)
        if not pruned_text.endswith('.'):
            pruned_text += '.'

        return pruned_text

# def encode(self, prompt):
#     return self.tokenizer.encode(prompt, return_tensors="pt")

# def decode(self, logits):
#     tok = torch.argmax(logits[:, -1, :]).item()
#     return self.tokenizer.decode(tok)

# def generate(self, model, x):
#     with torch.no_grad():
#         logits = model(x).logits
#     return logits