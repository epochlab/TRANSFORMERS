#!/usr/bin/env python3

import torch, chromadb
from transformers import AutoTokenizer, AutoModel

class vectorDB():
    def __init__(self, device):
        self.device = device
        print(self.device)
        self.client = chromadb.Client()
        self.memory = self.client.create_collection(name="vecdb")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)

    def push(self, data):
        uuid = str(self._length()).zfill(3)
        embedding = self._embed(data).cpu()
        self.memory.add(
            embeddings=[embedding[0].tolist()],
            documents=[data],
            ids=[uuid]
        )

    def pull(self, field):
        return self.memory.get(include=field)
    
    def _embed(self, x):
            tokens = self.tokenizer(x, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model(**tokens).last_hidden_state # [batch_size, sequence_length, hidden_size]
            return embedding.mean(dim=1) # Aggregate across sequence
    
    def _length(self):
        return self.memory.count()