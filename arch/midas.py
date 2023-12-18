#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image

model = DPTDepthModel(
        path=model_path,
        backbone="beitl16_512",
        non_negative=True,
    )
    net_w, net_h = 512, 512
    resize_mode = "minimal"
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

file = "/Users/James/Documents/test.png"

img = np.array(Image.open(file).convert('RGB'))
img = img.transpose((2, 0, 1))
sample = torch.tensor(img).float()
sample = sample.half()

pred = model.forward(sample.unsqueeze(0))

# def process(device, model, image):
#     sample = torch.from_numpy(image).to(device).unsqueeze(0)
#     sample = sample.half()

#     prediction = model.forward(sample)
#     prediction = (
#             torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=target_size[::-1],
#                 mode="bicubic",
#                 align_corners=False,
#                 )
#             .squeeze()
#             .cpu()
#             .numpy()
#         )

#     return prediction