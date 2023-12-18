#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.vectordb import vectorDB
from arch.utils import chat_playback

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

torch.manual_seed(123)

if __name__ == "__main__":
    db = vectorDB(DEVICE)
    LLM = Llama(MODEL_DIR, DEVICE)

    en1 = "Prometheus stole fire from the gods and gave it to man." 
    db.push(en1)

    tok = LLM.encode(en1)
    new_tok = LLM.generate(tok, max_length=128, temp=0.7)
    startpos = len(tok[0])
    response = LLM.decode(new_tok[:,startpos:-1])

    seq = ' '.join(response.split())
    print(seq)
    # en2 = LLM.prune(seq)
    # print(en2)
    # db.push(en2)

    # print(f"Prompt: {en1}")
    # print(f"Response: {en2}")
    # print(db.pull(['documents']))

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

# Build agent class
# Hide other Chroma fields (ID & Documents / Embeddings)