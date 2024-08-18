#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import torch
from arch.tokenizer import Tokenizer
from arch.model import Transformer
from arch.run import generate
from utils import device_mapper, load_config, albedo, chat_playback
from helper import Timing

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float16)
# torch.manual_seed(123)

MODEL_PATH = Path("/mnt/artemis/library/weights/meta/llama-2/7B-chat")

with Timing("Loading in "):
    tokenizer = Tokenizer(model_path=str(MODEL_PATH / "tokenizer.model"))
    transformer = Transformer.from_folder(MODEL_PATH, device=DEVICE, tokenizer=tokenizer)

print(transformer.params.max_seq_len)

print(f"Nparams: {sum(p.nelement() for p in transformer.parameters()):,}")

def main():
    system_call = load_config('profiles.yml')['diana']

    print("Use 'exit()' or press Ctrl-D to exit.")
    print("\nSystem Call:")
    print(f"{albedo(system_call, 'green')}")
    print("\n==================================\n")

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

    init_call = [B_SYS + system_call + E_SYS]
    init_toks = tokenizer.encode(' '.join(init_call), bos=True)
    log = init_call

    max_len = transformer.params.max_seq_len
    trunc_len = int(max_len*0.75)

    while True:
        prompt = input("User: ")
        if prompt.lower() == "exit()": break

        log += [B_INST + prompt + E_INST]

        toks = tokenizer.encode(' '.join(log), bos=True)
        
        if len(toks) + trunc_len >= max_len:
            toks = init_toks + toks[-trunc_len:]

        print(f"seq_len: {len(toks)}")
        new_toks, _ = generate(prompt_tokens=[toks], model=transformer, tokenizer=tokenizer, max_gen_len=None, temperature=0.7, top_p=0.9, logprobs=False)
        res = tokenizer.decode(new_toks[0]).strip()
        
        log += [res]

        chat_playback(f"> {res}")
        # print(albedo(log, "red")) # Print chat history

if __name__ == "__main__":
    main()