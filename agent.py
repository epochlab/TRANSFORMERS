#!/usr/bin/env python3

from pathlib import Path
from sentencepiece import SentencePieceProcessor

from arch.llm import Llama
from arch.vision import Vision
from arch.vectordb import VectorDB
from arch.utils import device_mapper, albedo, url2image

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

BLIP_PATH = "/Users/James/Documents/blip-image-captioning-large"
MODEL_DIR = "/Users/James/Documents/OpenHermes-2-Mistral-7B"
TOKENIZER_PATH = (MODEL_DIR if Path(MODEL_DIR).is_dir() else Path(MODEL_DIR).parent) + "/tokenizer.model"

def main():
    db = VectorDB(DEVICE)
    print(f"Signature: {db.client.heartbeat()}")

    tokenizer = SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    print(f"Vocab Size: {tokenizer.vocab_size()}")

    # LLM = Llama(MODEL_DIR, tokenizer, DEVICE)
    V1 = Vision(BLIP_PATH, DEVICE)

    en1 = "Prometheus stole fire from the gods and gave it to man."
    en2 = "Test ABC123"
    db.push([en1, en2])

    lookup = db.pull(str(000))['documents']
    print(lookup)

    # IM_START = tokenizer.bos_id()
    # IM_END = tokenizer.eos_id()

    # log = [IM_START] + tokenizer.encode(en) + [IM_END]
    # print(albedo(log, 'red')) # Print chat history
    # print(albedo(tokenizer.decode(log), "green"))

    # toks = LLM.encode(log)
    # new_tok = LLM.generate(toks, max_length=1024, temp=0.7)
    # response = LLM.decode(new_tok, skip_special=True)

    url = 'https://images.newscientist.com/wp-content/uploads/2017/02/08190042/gettyimages-615305114.jpg'
    raw_image = url2image(url)
    response = V1.witness(raw_image)

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

if __name__ == "__main__":
    main()