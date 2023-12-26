#!/usr/bin/env python3

from pathlib import Path
from sentencepiece import SentencePieceProcessor

from arch.utils import device_mapper
from models.llama import ModelArgs, Transformer

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

# "LLama2-7B": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
# 32 = n_emb - 2.484544 M parameters

MODEL_DIR = "/Users/James/Documents/OpenHermes-2-Mistral-7B"
TOKENIZER_PATH = (MODEL_DIR if Path(MODEL_DIR).is_dir() else Path(MODEL_DIR).parent) + "/tokenizer.model"

tokenizer = SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
print(f"Vocab Size: {tokenizer.vocab_size()}")

# model = Transformer(4096, 32, 32, 32000, 32).to(DEVICE)
# print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')