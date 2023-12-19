#!/usr/bin/env python3

import torch, random
from dataclasses import dataclass
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {str(DEVICE).upper()}")

MODEL_DIR = "/mnt/artemis/library/weights/stable-diffusion/sdxl/"

@dataclass
class HyperConfig:
    prompt = "a rural house on fire, cinematic, technicolor, film grain, analog, 70mm, 4K, IMAX"
    negative = """
            worst quality, low quality, lowres, low details, artifacts, cropped,
            oversaturated, undersaturated, overexposed, underexposed, letterbox, 
            aspect ratio, formatted, jpeg artifacts, draft, glitch, error,
            deformed, distorted, disfigured, duplicated, bad proportions,
            bad anatomy, bad eyes, closed eyes, missing fingers, missing hands
            """
    
    w, h = 1280, 544 # 2.35 Aspect ratio (1280x544, 1920x816, 2048x871)
    infer_steps = 100
    refine_steps = 150
    high_noise_fraction = 0.8
    cfg = 7.0
    seed = random.randint(0, 1e6)

config = HyperConfig()
print(f"Seed: {config.seed}")

vae = AutoencoderKL.from_single_file(
    MODEL_DIR + "sdxl_vae.safetensors",
    use_safetensors=True).to(DEVICE)

pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_DIR + "sd_xl_base_1.0.safetensors",
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    vae=vae)

# pipe.unfuse_lora()
# pipe.load_lora_weights(MODEL_DIR + "LoRA/JuggerCineXL2.safetensors")
# pipe.fuse_lora(lora_scale=0.5)
_ = pipe.to(DEVICE)

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
    MODEL_DIR + "sd_xl_refiner_1.0.safetensors",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    variant="fp16", 
    use_safetensors=True
    ).to(DEVICE)

# refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

generator = torch.Generator(device=DEVICE).manual_seed(config.seed)

latents = pipe(prompt=config.prompt,
               negative_prompt=config.negative,
               output_type='latent',
               height=config.h,
               width=config.w,
               num_inference_steps=config.infer_steps,
               denoising_end=config.high_noise_fraction,
               guidance_scale=config.cfg,
               generator=generator)

img = refiner(prompt=config.prompt,
              negative_prompt=config.negative,
              image=latents.images[0][None, :],
              num_inference_steps=config.refine_steps,
              denoising_start=config.high_noise_fraction,
              generator=generator)

img.images[0].save("test.png")