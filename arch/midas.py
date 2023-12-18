#!/usr/bin/env python3

from utils import img2tensor

file = "/Users/James/Documents/test.png"
sample = img2tensor(file)
print(sample.shape)

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