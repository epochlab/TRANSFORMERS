#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.utils import load_config, albedo, chat_playback, coder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    LLM = Llama(MODEL_DIR, DEVICE)
    
    system = load_config('profiles.yml')['jasmine']
    print(albedo(system['call'], 'green'))

    IM_START = LLM.tokenizer.bos_token
    IM_END = LLM.tokenizer.eos_token

    log = [IM_START] + [system['call']] + [IM_END]

    print("Type 'exit()' to cancel.")
    while True:
        x_inp = str(input("User: "))
        if x_inp == "exit()": break

        log += [IM_START] + [x_inp] + [IM_END]

        tok = LLM.encode(log)
        new_tok = LLM.generate(tok, max_length=1024, temp=0.7)
        response = LLM.decode(new_tok)

        log += [response.strip()]
        # print(albedo(log, 'red')) # Print chat history
        
        chat_playback(response)
        coder(response)

if __name__ == "__main__":
    main()