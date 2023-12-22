#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.vectordb import vectorDB
from arch.utils import load_config, albedo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    db = vectorDB(DEVICE)
    LLM = Llama(MODEL_DIR, DEVICE)

    system = load_config('profiles.yml')['jasmine']
    print(albedo(system['call'], 'green'))

    IM_START = LLM.tokenizer.bos_token
    IM_END = LLM.tokenizer.eos_token

    en1 = "How fast is a F16 jet?"
    db.push(en1)

    log = [IM_START] + [system['call']] + [IM_END] + [IM_START] + [en1] + [IM_END]
    print(albedo(log, 'red')) # Print chat history

    toks = LLM.encode(log)
    new_tok = LLM.generate(toks, max_length=1024, temp=0.7)
    response = LLM.decode(new_tok, skip_special=True)

    db.push(response)

    print(db.pull(['documents']))

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

if __name__ == "__main__":
    main()