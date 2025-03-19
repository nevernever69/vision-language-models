
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import os
from tqdm import tqdm
import random
from Model import tokenizer

from prep_data import class_names
import warnings
from Model import CLIP
from prep_data import share_dataset
warnings.filterwarnings('ignore')
emb_dim = 128
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
img_size = (80,80)
patch_size = (5,5)
n_channels = 3
vit_layers = 8
vit_heads = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text
vocab_size = 256
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 128
text_heads = 8
text_layers = 8
lr = 1e-3
epochs = 3
batch_size = 128
idx = 904

model = CLIP(emb_dim, vit_layers, vit_d_model, img_size,patch_size,n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers,text_d_model,retrieval = False).to(device)

_, val_dataset, _ = share_dataset()
model.load_state_dict(torch.load("clip.pt", map_location=device))

text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)

mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()])
mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

correct, total = 0,0
img = val_dataset[idx]["image"][None,:]

plt.imshow(img[0].permute(1, 2, 0)  ,cmap="gray")
plt.title(tokenizer(val_dataset[idx]["caption"], encode=False, mask=val_dataset[idx]["mask"][0])[0])
plt.show()
img = img.to(device)
with torch.no_grad():
  image_features = model.vision_encoder(img)
  text_features = model.text_encoder(text, mask=mask)


image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{class_names[int(index)]:>16s}: {100 * value.item():.2f}%")
