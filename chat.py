#!/usr/bin/env python3

from arch.llm import Llama
from arch.utils import device_mapper, load_config, albedo, chat_playback, coder

DEVICE = device_mapper()
print(f"Device: {str(DEVICE).upper()}")

def main():
    LLM = Llama(DEVICE)
    
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

        # url = 'https://images.newscientist.com/wp-content/uploads/2017/02/08190042/gettyimages-615305114.jpg'
        # raw_image = url2image(url)
        # response = V1.witness(raw_image)
        # print(response)

if __name__ == "__main__":
    main()