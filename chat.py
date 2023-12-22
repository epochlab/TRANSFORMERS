#!/usr/bin/env python3

import torch
from arch.llm import Llama
from arch.utils import load_config, chat_playback, coder, colored

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/mistral/OpenHermes-2-Mistral-7B"

def main():
    LLM = Llama(MODEL_DIR, DEVICE)
    
    profile = load_config('profiles.yml')['quentin']
    print(colored(profile['call'], 'green'))

    print("Type 'exit()' to cancel.")

    while True:
        x_inp = input("User: ")
        if x_inp == "exit()": break

        tok = LLM.encode(' '.join([profile['call'], x_inp]))
        new_tok = LLM.generate(tok, max_length=1024, temp=0.7)
        response = LLM.decode(new_tok)

        chat_playback(response)
        coder(response)

if __name__ == "__main__":
    main()