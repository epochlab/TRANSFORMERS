#!/usr/bin/env python3

import torch
from arch.llama import Llama
from arch.vectordb import vectorDB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(device).upper()}")

# torch.manual_seed(123)

if __name__ == "__main__":
    db = vectorDB(device)
    LLM = Llama()

    en1 = "Prometheus stole fire from the gods and gave it to man."  
    db.push(en1)

    seq = LLM.generate(en1)
    en2 = LLM.prune(seq)
    db.push(en2)

    print(f"Prompt: {en1}")
    print(f"Response: {en2}")
    print(db.pull(['documents']))

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

# Build agent class
# Hide other Chroma fields (ID & Documents / Embeddings)