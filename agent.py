#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.vectordb import vectorDB
from arch.utils import load_config, prune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    db = vectorDB(DEVICE)
    LLM = Llama(MODEL_DIR, DEVICE)

    profile = load_config('profiles.yml')['jasmine']

    en1 = "How fast does the earth spin?"
    db.push(en1)

    tok = LLM.encode(' '.join([profile['call'], en1]))
    new_tok = LLM.generate(tok, max_length=128, temp=0.7)
    startpos = len(tok[0])
    response = LLM.decode(new_tok[:,startpos:-1])

    en2 = prune(response)
    db.push(en2)

    print(db.pull(['documents']))

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

if __name__ == "__main__":
    main()