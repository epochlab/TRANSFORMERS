#!/usr/bin/env python3

from arch.llm import Llama
from arch.vision import Blip
from arch.vectordb import VectorDB
from arch.utils import device_mapper, load_config, albedo, url2image

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

def main():
    db = VectorDB(DEVICE)
    print(f"Signature: {db.client.heartbeat()}")

    LLM = Llama(DEVICE)
    # V1 = Blip(DEVICE)

    objective = "Calculate the speed difference between a F16 and the Japanese bullet train"

    state = None
    en = 'Construct a list of tasks.'
    db.push([en])

    project_agent = f"You are a superintelligent AI agent that considers the current objective {objective} and projects the next possible steps based on the previous engram or result {en}. You return short and precise results, with each entry less than 25 words. You ONLY respond to the current state {en} and do not overlap with: {state}."

    IM_START = LLM.tokenizer.bos_token
    IM_END = LLM.tokenizer.eos_token

    log = [IM_START] + [project_agent] + [IM_END]

    toks = LLM.encode(log)
    new_tok = LLM.generate(toks, max_length=1024, temp=0.7)
    response = LLM.decode(new_tok, skip_special=True)
    print(response)

    # url = 'https://images.newscientist.com/wp-content/uploads/2017/02/08190042/gettyimages-615305114.jpg'
    # raw_image = url2image(url)
    # response = V1.witness(raw_image)
    # print(response)

if __name__ == "__main__":
    main()