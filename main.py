#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Optional, Tuple, Literal, TypedDict

import torch
import torch.nn.functional as F

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
def generate(
    prompt_tokens: List[List[int]],
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    logprobs: bool = False,
    echo: bool = False,
) -> Tuple[List[List[int]], Optional[List[List[float]]]]:

    params = model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    if max_gen_len is None:
        max_gen_len = model.params.max_seq_len - 1

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
        logits = model.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(input=logits.transpose(1, 2), target=tokens, reduction="none", ignore_index=pad_id)

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
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

######

from arch.llm.tokenizer import Tokenizer
from arch.llm.model import ModelArgs, Transformer
from arch.utils import device_mapper, load_config, chat_playback, coder, albedo
from arch.helper import Timing

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

MODEL_PATH = "/mnt/artemis/library/weights/meta/llama-2/7Bf"
TOKENIZER_PATH = "/mnt/artemis/library/weights/meta/llama-2/tokenizer.model"

torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float16)
torch.manual_seed(1)

with Timing("Loading in "):
    with open(Path(MODEL_PATH) / "params.json", "r") as f:
        params = json.loads(f.read())

    tokenizer = Tokenizer(model_path=TOKENIZER_PATH)

    model_args: ModelArgs = ModelArgs(max_seq_len=512, max_batch_size=8, **params)
    model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args)
    checkpoint = torch.load(Path(MODEL_PATH) / "consolidated.00.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    print(model_args)
    print("Nparams:", sum(p.nelement() for p in model.parameters()))

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

Dialog = List[Message]
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

def main():
    system_call = load_config('profiles.yml')['jasmine']
    log: List[Dialog] = [[{"role": "system", "content": system_call}]]

    print("Use 'exit()' or press Ctrl-D to exit.")
    print(f"{albedo(system_call, 'green')}")
    print("\n==================================\n") 

    while True:
        prompt = input("User: ")
        if prompt.lower() == "exit()": break
        log[-1].append({"role": "user", "content": prompt})

        dialog = log[-1]
        if dialog[0]["role"] == "system":
            dialog = [{"role": dialog[1]["role"], "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"]}] + dialog[2:]

        dialog_toks: List[int] = sum([tokenizer.encode(f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()}") for prompt, answer in zip(dialog[::2], dialog[1::2],)],[],)
        dialog_toks += tokenizer.encode(f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}")

        toks = [dialog_toks[-256:]]

        with Timing("Total: ", enabled=False, on_exit=lambda x: f", {1e9/x:.2f} tok/sec"):
            logprobs = False
            new_toks, tok_logprobs = generate(prompt_tokens=toks, max_gen_len=1024, temperature=0.6, top_p=0.9, logprobs=logprobs)

        if logprobs:
            result = [{"generation": {"role": "assistant", "content": tokenizer.decode(t)}, "tokens": [tokenizer.decode(x) for x in t], "logprobs": logprobs_i,} for t, logprobs_i in zip(new_toks[0], tok_logprobs)] ### Fix
        else:
            result = {"generation": {"role": "assistant", "content": tokenizer.decode(new_toks[0])}}

        log[-1].append(result['generation'])
        # print(albedo(log, "red")) # Print chat history

        chat_playback(f"> {result['generation']['content']}")
        coder(result['generation']['content'])

if __name__ == "__main__":
    main()