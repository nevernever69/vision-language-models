# Load the dataset
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
df = pd.read_csv('fashion/myntradataset/styles.csv', usecols=['id',  'subCategory'])

unique, counts = np.unique(df["subCategory"].tolist(), return_counts = True)
print(f"Classes: {unique}: {counts}")

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)

# Print the sizes of the datasets
print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
class_names = df['subCategory'].unique()
class_names = [str(name).lower() for name in class_names]

# Replace in-place
for i, name in enumerate(class_names):
    if name == "lips":
        class_names[i] = "lipstick"
    elif name == "eyes":
        class_names[i] = "eyelash"
    elif name == "nails":
        class_names[i] = "nail polish"

captions = {idx: class_name for idx, class_name in enumerate(class_names)}

for idx, caption in captions.items():
    print(f"{idx}: {caption}\n")

class MyntraDataset(Dataset):
    def __init__(self, data_frame, captions, target_size=28):

        self.data_frame = data_frame[data_frame['subCategory'].str.lower() != 'innerwear']
        self.target_size = target_size  # Desired size for the square image
        self.transform = T.Compose([
            T.ToTensor()  # Convert image to tensor
        ])

        self.captions = captions

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        while True:
            sample = self.data_frame.iloc[idx]
            img_path = os.path.join("fashion/myntradataset/images", f"{sample['id']}.jpg")

            try:
                # Attempt to open the image
                image = Image.open(img_path).convert('RGB')
            except (FileNotFoundError, IOError):
                # If the image is not found, skip this sample by incrementing the index
                idx = (idx + 1) % len(self.data_frame)  # Loop back to the start if we reach the end
                continue  # Retry with the next index

            # Resize the image to maintain aspect ratio
            image = self.resize_and_pad(image, self.target_size)

            # Apply transformations (convert to tensor)
            image = self.transform(image)

            # Retrieve the subCategory label and its corresponding caption
            label = sample['subCategory'].lower()
            label = {"lips": "lipstick", "eyes": "eyelash", "nails": "nail polish"}.get(label, label)

            label_idx = next(idx for idx, class_name in self.captions.items() if class_name == label)

            # # Tokenize the caption using the tokenizer function
            cap, mask = tokenizer(self.captions[label_idx])

            # Make sure the mask is a tensor
            mask = torch.tensor(mask)

            # If the mask is a single dimension, make sure it is expanded correctly
            if len(mask.size()) == 1:
                mask = mask.unsqueeze(0)

            return {"image": image, "caption": cap, "mask": mask,"id": img_path}

    def resize_and_pad(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        image = image.resize((new_width, new_height))

        pad_width = (target_size - new_width) // 2
        pad_height = (target_size - new_height) // 2

        padding = (pad_width, pad_height, target_size - new_width - pad_width, target_size - new_height - pad_height)
        image = ImageOps.expand(image, padding, fill=(0, 0, 0))

        return image

#Create datasets
def share_dataset():
    
    train_dataset = MyntraDataset(data_frame=train_df ,captions = captions, target_size =80)
    val_dataset = MyntraDataset(data_frame=val_df ,captions = captions ,target_size =80)
    test_dataset = MyntraDataset(data_frame=val_df, captions = captions, target_size = 224)
    return train_dataset, val_dataset, test_dataset
def prepare_data(batch_size):
    train_dataset, val_dataset, test_dataset = share_dataset()

    print("Number of Samples in Train Dataset:", len(train_dataset))
    print("Number of Samples in Validation Dataset:", len(val_dataset))


    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size,num_workers = 5)
    val_loader  = DataLoader(val_dataset, shuffle = False, batch_size = batch_size,num_workers = 5)
    test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 5)

    #Sanity check of dataloader initialization
    len(next(iter(train_loader)))  #(img_tensor,label_tensor)
    return train_loader, val_loader, test_loader

