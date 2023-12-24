#!/usr/bin/env python3

from arch.llm import Llama
from arch.vision import Vision
from arch.vectordb import VectorDB
from arch.utils import device_mapper, load_config, albedo, url2image

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

BLIP = "/Users/James/Documents/blip-image-captioning-large/"
MISTRAL = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    db = vectorDB(DEVICE)
    LLM = Llama(MISTRAL, DEVICE)
    v1 = Vision(BLIP, DEVICE)

    system = load_config('profiles.yml')['creation']
    print(albedo(system['call'], 'green'))

    # Replace with token_wrapper
    IM_START = LLM.tokenizer.bos_token
    IM_END = LLM.tokenizer.eos_token

    en = "How to bake a loaf of bread?"
    db.push(en)

    log = [IM_START] + [system['call']] + [IM_END] + [IM_START] + [en] + [IM_END] # <- Script???
    print(albedo(log, 'red')) # Print chat history

    toks = LLM.encode(log)
    new_tok = LLM.generate(toks, max_length=1024, temp=0.7)
    response = LLM.decode(new_tok, skip_special=True)

    db.push(response)
    print(db.pull(['documents']))

    url = 'https://hips.hearstapps.com/hmg-prod/images/is-santa-claus-real-1632187377.jpg?'
    raw_image = url2image(url)
    print(v1.witness(raw_image))

    # index = faiss.IndexFlatL2(embedding.shape[1])
    # index.add(embedding.numpy())
    # print(index.reconstruct[1])

if __name__ == "__main__":
    main()