#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

class Llama():
    def __init__(self, device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def generate(self, tok, max_length, temp):
        attn_mask = torch.tensor([1] * len(tok[0])).unsqueeze(0).to(self.device)
        
        tok_length: int = 0
        while tok_length <= 2:
            new_tok = self.model.generate(
                input_ids = tok,
                attention_mask = attn_mask,
                max_new_tokens = max_length,
                do_sample = True,
                top_p = 0.92,
                temperature = temp,
                pad_token_id = self.tokenizer.pad_token_id,)
            
            # new_tok = torch.cat((new_tok[:,0].reshape(1,1), new_tok[:,len(tok[0]):]), dim=1)
            new_tok = new_tok[:,len(tok[0]):]
            tok_length = new_tok.shape[1]

        return new_tok

    def encode(self, inp):
        return self.tokenizer(' '.join(inp), return_tensors="pt").to(self.device).input_ids

    def decode(self, toks, skip_special=False):
        return self.tokenizer.batch_decode(toks, skip_special_tokens=skip_special, clean_up_tokenization_spaces=True)[0]
    
    def token_wrapper(self, tok):
        return [self.tokenizer.bos_token, tok, self.tokenizer.eos_token]