#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

class Llama():
    def __init__(self, model_dir, device):
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                          torch_dtype=torch.float16, 
                                                          device_map="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='left')
        self.device = device

    def generate(self, tok, max_length, temp):
        attn_mask = torch.tensor([1] * len(tok[0])).unsqueeze(0).to(self.device)

        tok_length: int = 0
        while tok_length <= 2:
            new_tok = self.model.generate(
                input_ids = tok,
                attention_mask = attn_mask,
                max_new_tokens = max_length,
                do_sample = True,
                top_p = 1.0,
                top_k = 10,
                temperature = temp,
                pad_token_id = self.tokenizer.pad_token_id,
                repetition_penalty = 1.0,
                length_penalty = 1.0)
            
            new_tok = torch.cat((new_tok[:,0].reshape(1,1), new_tok[:,len(tok[0]):]), dim=1)
            tok_length = new_tok.shape[1]

        return new_tok

    def encode(self, prompt):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids

    def decode(self, tok):
        return self.tokenizer.batch_decode(tok, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]