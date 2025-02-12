import io

import vit_prisma

import numpy as np
import matplotlib.pyplot as plt

import json
from IPython.core.display import display, HTML
import copy
from tqdm import tqdm
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import datasets, transforms

# 数据集根目录
dataset_root = '/home/gated/Downloads/mindataset'
from torch.utils.data import Subset
import timm


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
batch_size = 128
# testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
testset = datasets.ImageFolder(root=dataset_root, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
from vit_prisma.utils.get_activations import timmCustomAttention # Custom Attention has hook functions

# Load original model  VIT模型的网络在block中  该模型有12个块
orig_model = timm.models.create_model('vit_base_patch32_224', pretrained=True,pretrained_cfg_overlay=dict(file='/home/gated/captum/vit_models/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'))
print(orig_model)
num_blocks = len(orig_model.blocks)
print(f"Number of Blocks: {num_blocks}")


# 获取模型的维度信息，通常是指嵌入向量的维度
# 这个信息位于模型的embedding层或者是其transformer block的一部分
embed_dim = orig_model.embed_dim
print(f"Embedding Dimension: {embed_dim}")
# Replace original model's attention with hooked attention
model = copy.deepcopy(orig_model)
for idx, block in enumerate(model.blocks):
  model.blocks[idx].attn = timmCustomAttention(dim=embed_dim,num_heads=num_blocks,qkv_bias=True)

# Reset attention with pretrained weights from original model.
model.load_state_dict(orig_model.state_dict())

from vit_prisma.utils.get_activations import get_activations

# Get all attention patterns across test dataset   将模型，层，数据集分别激活  得到层注意力列表
total_activations = []
from vit_prisma.utils.get_activations import get_activations

# Get all attention patterns across test dataset
for idx, block in enumerate(model.blocks):
    print("On block {}".format(idx))
    activations = get_activations(model, model.blocks[idx].attn.attn_scores, testloader, use_cuda=True)
    total_activations.append(activations)

print("Shape of activations per layer: {}".format(activations.shape))

# Choose one image sample to examine
datapoint_idx = 1

# Plot image
image, label = testset[datapoint_idx]
image = image.permute(1, 2, 0)
image = image.numpy()
plt.imshow(image)
plt.title(f"Label: {label}")
plt.show()


from vit_prisma.visualization.visualize_attention import plot_attn_heads

# Plot all attention head activations for your chosen image
plot_attn_heads(total_activations, idx=datapoint_idx,
                n_heads = 12, n_layers = 12, img_shape=50, figsize=(20,20),
                global_min_max=True, global_normalize=False, fourier_transform_local=False,
                graph_type = "imshow_graph")

# transformer头的均值和方差
total_means = []
total_var = []
for i in range(12):
    for j in range(12):
        head_data = total_activations[i][:,j,:,:].reshape(batch_size, -1)
        m = np.mean(head_data, axis=0)
        v = np.var(head_data, axis=0)
        total_means.append(np.mean(m, axis=0))
        total_var.append(np.mean(v, axis=0))


labels = []
for i in range(12):
    for j in range(12):
        labels.append(f"Layer {i}, Head {j}")

total_stds = np.sqrt(total_var)

# Plotting the flattened Silhouette scores
plt.figure(figsize=(20, 6))
plt.bar(range(len(total_means)), total_means, tick_label=labels, yerr=total_stds, capsize=2)
plt.xticks(rotation=90, fontsize=7)  # Rotate x labels for better visibility if they're long
plt.xlabel('Attention Head Index')
plt.ylabel('Mean')
plt.title('Means of Attention Scores by Head Across Test Set')
plt.show()
#经过傅里叶变换的注意力头的均值和方差
total_means = []
total_var = []
for i in range(12):
    for j in range(12):
        head_data = total_activations[i][:,j,:,:]
        head_data = np.abs(np.fft.fft2(head_data)).reshape(batch_size, -1)
        m = np.mean(head_data, axis=0)
        v = np.var(head_data, axis=0)
        total_means.append(np.mean(m, axis=0))
        total_var.append(np.mean(v, axis=0))

# Create labels for the attention heads
labels = []
for i in range(12):
    for j in range(12):
        labels.append(f"Layer {i}, Head {j}")

total_stds = np.sqrt(total_var)

# Plotting the flattened Silhouette scores
plt.figure(figsize=(20, 6))
plt.bar(range(len(total_means)), total_means, tick_label=labels, yerr=total_stds, capsize=2)
plt.xticks(rotation=90, fontsize=7)  # Rotate x labels for better visibility if they're long
plt.xlabel('Attention Head Index')
plt.ylabel('Mean')
plt.title('Means of Fourier-Transformed Attention Scores by Head Across Test Set')
plt.show()

from vit_prisma.visualization.visualize_attention_js import plot_javascript
from IPython.core.display import display, HTML

layer_idx = 0
datapoint_idx = 2
attn_head_idx = 0

attn_head = total_activations[layer_idx][datapoint_idx][attn_head_idx]
html_code = plot_javascript(attn_head, testset[datapoint_idx][0], ATTN_SCALING=8, cls_token=True)
from IPython.core.display import display, HTML
display(HTML(html_code))

# ----------------------------------Kmeans算法 注意力头的Silhouette分数
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_silhouette_scores(total_activations, layer_num, head_num, n_clusters=5, fourier=False):
    activations = total_activations[layer_num][:, head_num, :, :]
    if fourier:
        activations = np.abs(np.fft.fft2(activations))
    flattened_activations = activations.reshape(10000, -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flattened_activations)
    score = silhouette_score(flattened_activations, kmeans.labels_)
    return score
# 从Kmeans中获取分数
total_silhouette_scores = []
for i in range(12):
    for j in range(12):
        silhouette_scores = get_silhouette_scores(total_activations, i, j)
        total_silhouette_scores.append(silhouette_scores)

# Create labels for the attention heads
labels = []
for i in range(12):
    for j in range(12):
        labels.append(f"Layer {i}, Head {j}")

# Plotting the flattened Silhouette scores
plt.figure(figsize=(20, 6))
plt.bar(range(len(total_silhouette_scores)), total_silhouette_scores, tick_label=labels)
plt.xticks(rotation=90, fontsize=7)  # Rotate x labels for better visibility if they're long
plt.xlabel('Attention Head Index')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores of Attention Heads')
plt.show()

# Get top heads by silhouette score    打印每一个层的最重要注意力头
def get_layer_head(idx):
    layer_num = idx // 12
    head_num = idx % 12
    return layer_num, head_num

top_heads = np.argsort(total_silhouette_scores)[::-1][:10]

print("The top heads by Silhouette Score are:")
for head in top_heads:
    layer_num, head_num = get_layer_head(head)
    print(f"Layer {layer_num}, Head {head_num}")


avg_frobenius_norms_per_head = []
var_frobenius_norms_per_head = []

for layer in total_activations:
    norms = np.linalg.norm(layer, ord='fro', axis=(-2, -1))
    avg_norm_per_head = np.mean(norms, axis=0)
    var_norm_per_head = np.var(norms, axis=0)
    avg_frobenius_norms_per_head.append(avg_norm_per_head)
    var_frobenius_norms_per_head.append(var_norm_per_head)

avg_frobenius_norms_per_head = np.array(avg_frobenius_norms_per_head)
var_frobenius_norms_per_head = np.array(var_frobenius_norms_per_head)


# plot bar plot of frobrenius norms by attentoin head
# Create labels for the attention heads

labels = []
for i in range(12):
    for j in range(12):
        labels.append(f"Model {i} - Head {j}")

flattened_fn = avg_frobenius_norms_per_head.flatten()
std_frobenius_norms_per_head = np.sqrt(var_frobenius_norms_per_head.flatten())

plt.figure(figsize=(20, 6))
plt.bar(range(len(flattened_fn)), flattened_fn, tick_label=labels, yerr=std_frobenius_norms_per_head, capsize=2)
plt.xticks(rotation=90, fontsize=7)  # Rotate x labels for better visibility if they're long
plt.xlabel('Attention Head Index')
plt.ylabel('Average Frobrenius Norm')
plt.title('Average Frobrenius Norms of Attention Heads')
plt.show()

# sort histograms by frobrenius norm

# Idx to layer, head number
def get_layer_head(idx):
    layer_num = idx // 12
    head_num = idx % 12
    return layer_num, head_num

# Get top heads
top_heads = np.argsort(avg_frobenius_norms_per_head.flatten())[-10:][::-1]
print("The top heads by Frobrenius Norm are:")
for head in top_heads:
    layer_num, head_num = get_layer_head(head)
    print(f"Layer {layer_num}, Head {head_num}")