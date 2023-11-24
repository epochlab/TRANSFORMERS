#!/usr/bin/env python3

# import faiss
import torch, chromadb
import transformers
from transformers import AutoTokenizer, AutoModel, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class vectorDB():
    def __init__(self):
        self.client = chromadb.Client()
        self.memory = self.client.create_collection(name="vecdb")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    def _embed(self, x):
            tokens = self.tokenizer(x, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = self.model(**tokens).last_hidden_state # [batch_size, sequence_length, hidden_size]
            return embedding.mean(dim=1) # Aggregate across sequence

    def _add(self, data, embedding):
        self.memory.add(
            embeddings=[embedding[0].tolist()],
            documents=[data],
            ids=[str(0)]
        )

class Llama():
    def __init__(self):
        self.model = "/mnt/artemis/library/weights/meta/llama-2/7Bf/hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )

        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

db = vectorDB()
LLM = Llama()
LLM.generate('I liked "Fight Club" and "Jaws". Do you have any recommendations of other films I might like?\n')

en1 = "Prometheus stole fire from the gods and gave it to man."
embedding = db._embed(en1)
db._add(en1, embedding.cpu())

print(db.memory.get(include=['embeddings', 'documents']))

# index = faiss.IndexFlatL2(embedding.shape[1])
# index.add(embedding.numpy())
# print(index.reconstruct[1])