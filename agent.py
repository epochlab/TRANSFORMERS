#!/usr/bin/env python3

# import faiss
import torch, chromadb
import transformers
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(device).upper()}")

# torch.manual_seed(123)

class vectorDB():
    def __init__(self):
        self.client = chromadb.Client()
        self.memory = self.client.create_collection(name="vecdb")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    def push(self, data):
        uuid = str(self._length()).zfill(3)
        embedding = db._embed(data).cpu()
        self.memory.add(
            embeddings=[embedding[0].tolist()],
            documents=[data],
            ids=[uuid]
        )

    def pull(self, field):
        return self.memory.get(include=field)
    
    def _embed(self, x):
            tokens = self.tokenizer(x, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = self.model(**tokens).last_hidden_state # [batch_size, sequence_length, hidden_size]
            return embedding.mean(dim=1) # Aggregate across sequence
    
    def _length(self):
        return self.memory.count()

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
        return sequences

    def prune(self, seq):
        if not seq:
            raise ValueError("Error: Empty sequence generated.")

        text = seq[0]['generated_text'].replace('\n', ' ')
        sentences = [sentence for sentence in text.split('. ') if sentence.strip()][1:]

        if sentences and not text.strip().endswith('.'):
            sentences = sentences[:-1]

        pruned_text = '. '.join(sentences)
        if not pruned_text.endswith('.'):
            pruned_text += '.'

        return pruned_text

if __name__ == "__main__":
    db = vectorDB()
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