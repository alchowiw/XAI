from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
from data.imagenet_utils import CLS2IDX
use_thresholding =  False



normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# initialize ViT pretrained
model = vit_LRP(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)

def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def print_top_classes(predictions, **kwargs):    
  # Print Top-5 predictions
  prob = torch.softmax(predictions, dim=1)
  class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
  max_str_len = 0
  class_names = []
  for cls_idx in class_indices:
      class_names.append(CLS2IDX[cls_idx])
      if len(CLS2IDX[cls_idx]) > max_str_len:
          max_str_len = len(CLS2IDX[cls_idx])

  print('Top 5 classes:')
  for cls_idx in class_indices:
      output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
      output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
      output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
      print(output_string)
  return class_indices,class_names

import os
import matplotlib.pyplot as plt
from PIL import Image

input_dir = '/media/gated/3d8e13a5-e3e0-4c1b-bbdb-0cf8821ad637/imagenet'  # 输入的文件夹路径，如 'samples/'
output_dir = '/media/gated/3d8e13a5-e3e0-4c1b-bbdb-0cf8821ad637/imagenet_out'  # 输出的文件夹路径，如 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
image = Image.open('samples/catdog.png')

dog_cat_image = transform(image)
print("---------------------------",dog_cat_image.shape)
output = model(dog_cat_image.unsqueeze(0).cuda())
indices_list,name_list = print_top_classes(output)
print(indices_list,'/n',name_list)

fig, axs = plt.subplots(1, 6)
axs[0].imshow(image)
axs[0].axis('off')

idx = 0
for i in indices_list:
# generate visualization for class 243: 'bull mastiff'
    print(i)
    pic = generate_visualization(dog_cat_image, class_index=indices_list[idx])
    
    axs[idx+1].imshow(pic)
    axs[idx+1].axis('off')
    axs[idx+1].set_title(name_list[idx])
    idx+=1

plt.savefig('out.png')
plt.show()





# 遍历文件夹
for file_name in os.listdir(input_dir):
    # 跳过非 png 文件
    if not file_name.endswith('.png'): 
        continue
    
    image_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, 'out_' + file_name)

    # 加载图片
    image = Image.open(image_path)
    dog_cat_image = transform(image)
    output = model(dog_cat_image.unsqueeze(0).cuda())
    indices_list, name_list = print_top_classes(output)

    fig, axs = plt.subplots(1, 6)
    axs[0].imshow(image)
    axs[0].axis('off')

    idx = 0
    for i in indices_list:
        # 生成可视化结果
        pic = generate_visualization(dog_cat_image, class_index=indices_list[idx])
    
        axs[idx+1].imshow(pic)
        axs[idx+1].axis('off')
        axs[idx+1].set_title(name_list[idx])
        idx+=1

    # 保存输出图片
    fig.savefig(output_path)
    plt.show()

# prev_use_thresholding = use_thresholding
# if not use_thresholding:
#   use_thresholding = True
# image = Image.open('samples/catdog.png')
# dog_cat_image = transform(image)

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(image)
# axs[0].axis('off')

# output = model(dog_cat_image.unsqueeze(0).cuda())
# print_top_classes(output)

# # cat - the predicted class
# cat = generate_visualization(dog_cat_image)

# # dog 
# # generate visualization for class 243: 'bull mastiff'
# dog = generate_visualization(dog_cat_image, class_index=243)

# if not prev_use_thresholding:
#   use_thresholding = False

# axs[1].imshow(cat)
# axs[1].axis('off')
# axs[2].imshow(dog)
# axs[2].axis('off')
# plt.savefig('2.png')
# plt.show()

