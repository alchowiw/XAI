import numpy as np
import torch
from PIL import Image
import os.path
import argparse
from pathlib import Path
import cv2
import heapq
from torch.nn import functional as F
from torch.utils.data import DataLoader
import tqdm
import einops
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from utils.factory import create_model_and_transforms, get_tokenizer
from utils.visualization import image_grid, visualization_preprocess
from prs_hook import hook_prs_logger
from matplotlib import pyplot as plt


device = 'cuda:0'
pretrained = '/media/3d8e13a5-e3e0-4c1b-bbdb-0cf8821ad637/openclip_model/ViT-H-14.bin' # 'laion2b_s32b_b79k'
model_name = 'ViT-H-14' 



model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
model.to(device)
model.eval()
context_length = model.context_length
vocab_size = model.vocab_size
tokenizer = get_tokenizer(model_name)

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)
print("Len of res:", len(model.visual.transformer.resblocks))

prs = hook_prs_logger(model, device)

image_pil = Image.open('images/catdog.png')
image = preprocess(image_pil)[np.newaxis, :, :, :]
_ = plt.imshow(image_pil)

prs.reinit()
with torch.no_grad():
    representation = model.encode_image(image.to(device), 
                                        attn_method='head', 
                                        normalize=False)
    attentions, mlps = prs.finalize(representation)  # attentions: [1, 32, 257, 16, 1024], mlps: [1, 33, 1024]

# lines = ['An image of a dog', 'An image of a cat','an image of a lion']
# texts = tokenizer(lines).to(device)  # tokenize
# class_embeddings = model.encode_text(texts)
# class_embedding = F.normalize(class_embeddings, dim=-1)

# attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T

# # An image of a dog:
# attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
#                                   scale_factor=model.visual.patch_size[0],
#                                   mode='bilinear').to(device)
# attention_map = attention_map[0].detach().cpu().numpy()
# print(attention_map)
# print(lines[0])
# plt.imshow(attention_map[0] - np.mean(attention_map,axis=0))

# v = attention_map[0] - attention_map[1] # np.mean(attention_map,axis=0)
# min_ = min((attention_map[0] - attention_map[1]).min(), (attention_map[1] - attention_map[0]).min())
# max_ = max((attention_map[0] - attention_map[1]).max(), (attention_map[1] - attention_map[1]).max())
# v = v - min_
# v = np.uint8((v / (max_-min_))*255)
# high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
# print(high)
# plt.colorbar()
# plt.axis('off')
# plt.show()
# print(lines[1])
# plt.imshow(attention_map[1] - np.mean(attention_map,axis=0),)

# v = attention_map[1] - attention_map[0]
# v = v - min_
# v = np.uint8((v / (max_-min_))*255)
# high = cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
# print(high)
# plt.colorbar()
# plt.axis('off')
# plt.show()


# lines = ['An image of a dog', 'An image of a cat', 'An image of a bird']  # 添加第三个文本
# texts = tokenizer(lines).to(device)  # tokenize
# class_embeddings = model.encode_text(texts)
# class_embedding = F.normalize(class_embeddings, dim=-1)

# attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
# attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
#                                   scale_factor=model.visual.patch_size[0],
#                                   mode='bilinear').to(device)
# attention_map = attention_map[0].detach().cpu().numpy()

# for i, line in enumerate(lines):
#     print(line)
#     plt.imshow(attention_map[i] - np.mean(attention_map,axis=0))
#     plt.colorbar()
#     plt.axis('off')
#     plt.show()


lines = ['An image of a dog', 'An image of a cat', 'An image of a bird']  # 添加第三个文本
texts = tokenizer(lines).to(device)  # tokenize
class_embeddings = model.encode_text(texts)
class_embedding = F.normalize(class_embeddings, dim=-1)

attention_map = attentions[0, :, 1:, :].sum(axis=(0,2)) @ class_embedding.T
attention_map = F.interpolate(einops.rearrange(attention_map, '(B N M) C -> B C N M', N=16, M=16, B=1), 
                                  scale_factor=model.visual.patch_size[0],
                                  mode='bilinear').to(device)
attention_map = attention_map[0].detach().cpu().numpy()
print(attention_map)
fig, axs = plt.subplots(1, len(lines), figsize=(15,5))  # 创建一个1行，len(lines)列的子图

for i, line in enumerate(lines):
    if len(lines)==1:
        im = axs[i].imshow(attention_map[i]) 
        axs[i].axis('off')
        axs[i].title.set_text(line)
    else:
        im = axs[i].imshow(attention_map[i] - np.mean(attention_map,axis=0)) 
        axs[i].axis('off')
        axs[i].title.set_text(line)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()


