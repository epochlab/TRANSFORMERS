#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from arch.tokenizer import Tokenizer
from arch.llama.model import ModelArgs, Transformer
from utils import device_mapper, load_config, albedo, chat_playback
from helper import Timing

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float16)

MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
TOKENIZER_PATH = "/mnt/artemis/library/weights/meta/llama-2/tokenizer.model"

with Timing("Loading in "):
    with open(Path(MODEL_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

    args: ModelArgs = ModelArgs(max_seq_len=512, max_batch_size=8, **params)
    args.vocab_size = tokenizer.n_words

    transformer = Transformer(args)
    ckpt = torch.load(Path(MODEL_PATH) / "consolidated.00.pth", map_location="cpu")
    transformer.load_state_dict(ckpt, strict=False)

    print(args)
    print(f"Nparams: {sum(p.nelement() for p in transformer.parameters()):,}")

def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)

def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)

@torch.inference_mode()
def generate(prompt_tokens: List[List[int]], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9, logprobs: bool = False, echo: bool = False,) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
    params = transformer.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    if max_gen_len is None:
        max_gen_len = transformer.params.max_seq_len - 1

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=DEVICE)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=DEVICE)
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=DEVICE)
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = transformer.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(input=logits.transpose(1, 2), target=tokens, reduction="none", ignore_index=pad_id)

    for cur_pos in range(min_prompt_len, total_len):
        logits = transformer.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        if logprobs:
            token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens[:, prev_pos + 1 : cur_pos + 1],
                reduction="none",
                ignore_index=pad_id)
        
        eos_reached |= (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
        prev_pos = cur_pos
        if all(eos_reached): break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if tokenizer.eos_id in toks:
            eos_idx = toks.index(tokenizer.eos_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)
    return (out_tokens, out_logprobs if logprobs else None)

def main():
    system_call = load_config('profiles.yml')['jasmine']

    print("Use 'exit()' or press Ctrl-D to exit.")
    print(f"{albedo(system_call, 'green')}")
    print("\n==================================\n")

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"    

    log = [B_SYS + system_call + E_SYS]

    while True:
        prompt = input("User: ")
        if prompt.lower() == "exit()": break

        log += [B_INST + prompt + E_INST]

        toks = tokenizer.encode(' '.join(log), bos=True)
        new_toks, _ = generate(prompt_tokens=[toks], max_gen_len=None, temperature=0.7, top_p=0.9, logprobs=False)
        res = tokenizer.decode(new_toks[0]).strip()

        log += [res]

        chat_playback(f"> {res}")
        # print(albedo(log, "red")) # Print chat history

if __name__ == "__main__":
    main()