#!/usr/bin/env python3

import torch, chromadb
from transformers import AutoTokenizer, AutoModel

class VectorDB():
    def __init__(self, device):
        self.device = device
        self.client = chromadb.Client()
        self.memory = self.client.create_collection(name="vecdb")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.padding: int = 3

    def push(self, data):
        for sample in data:
            prune = sample.strip()
            uuid = str(self._length()).zfill(self.padding)
            embedding = self._embed(prune).cpu()
            self.memory.add(
                embeddings=[embedding[0].tolist()],
                documents=[prune],
                ids=[uuid])

    def pull(self, uuid=None):
        if uuid != None: return self.memory.get(ids=[uuid.zfill(self.padding)], include=['embeddings' ,'documents'])
        else: return self.memory.get(include=['embeddings' ,'documents'])
    
    def delete(self, uuid):
        return self.memory.delete(ids=[uuid.zfill(self.padding)])
    
    def query():
        return None

    def _embed(self, x):
            tokens = self.tokenizer(x, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embedding = self.model(**tokens).last_hidden_state # [batch_size, sequence_length, hidden_size]
            return embedding.mean(dim=1) # Aggregate across sequence
    
    def _length(self):
        return self.memory.count()