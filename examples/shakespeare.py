#!/usr/bin/env python3

import sys
sys.path.append('..')

import torch
from models.transformer import Transformer
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(device).upper()}")

#DATA
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#HYPERPARAMETERS
@dataclass
class HyperConfig:
    batch_size = 16 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
    cfg = 7.0

    max_iters = 5000
    eval_interval = 100
    learning_rate = 1e-3
    eval_iters = 200

    # Unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

config = HyperConfig()
print(config.vocab_size)

stoi = { ch:i for i,ch in enumerate(config.chars) }
itos = { i:ch for i,ch in enumerate(config.chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% = Train | 10% = Test
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#CONSTRUCT MODEL
model = Transformer(config.n_embd, config.n_head, config.n_layer, config.vocab_size, config.block_size).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

#TRAIN
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
for iter in range(config.max_iters):
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#INFERENCE
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))