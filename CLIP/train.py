
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
from Model import CLIP
from prep_data import prepare_data
class Trainer:
    def __init__(self, model, optimizer,  device):
        self.model = model
        self.optimizer = optimizer
        # self.loss = loss_fn
        self.device = device

    def train(self, train_loader, test_loader, epochs, save_model_every_n_epochs=1):
        best_loss = np.inf
        for epoch in range(epochs):
            epoch_loss = 0.0  # To accumulate the loss over the epoch
            with tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]") as tepoch:
                for i, data in tepoch:
                    img, cap, mask = data["image"].to(self.device), data["caption"].to(self.device), data["mask"].to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.model(img, cap, mask)
                    loss.backward()
                    self.optimizer.step()
                    # Update the progress bar with the current loss
                    tepoch.set_postfix(loss=loss.item())
                    epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.3f}")

            # Save model if it performed better than the previous best
            if avg_loss <= best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), "clip.pt")
                print("Model Saved.")
    def test(self, testloader):
        model = CLIP(emb_dim, vit_layers, vit_d_model, img_size,patch_size,n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers,text_d_model,retrieval = False).to(device)

        model.load_state_dict(torch.load("clip.pt", map_location=device))

        text = torch.stack([tokenizer(x)[0] for x in val_dataset.captions.values()]).to(device)

        mask = torch.stack([tokenizer(x)[1] for x in val_dataset.captions.values()])
        mask = mask.repeat(1,len(mask[0])).reshape(len(mask),len(mask[0]),len(mask[0])).to(device)

        correct, total = 0,0
        with torch.no_grad():
            for data in val_loader:

                images, labels = data["image"].to(device), data["caption"].to(device)
                image_features = model.vision_encoder(images)
                text_features = model.text_encoder(text, mask=mask)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                _, indices = torch.max(similarity,1)

                pred = torch.stack([tokenizer(val_dataset.captions[int(i)])[0] for i in indices]).to(device)
                correct += int(sum(torch.sum((pred==labels),dim=1)//len(pred[0])))

                total += len(labels)

        print(f'\nModel Accuracy: {100 * correct // total} %')

    def save_model(self, epoch):
        model_path = f"models/{self.exp_name}_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

def main():
    # Training parameters
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

    print(f"Starting experiment: {epochs} epochs")
    # Load the CIFAR10 dataset
    train_loader, val_loader, test_loader= prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = CLIP(emb_dim, vit_layers, vit_d_model, img_size,patch_size,n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers,text_d_model, retrieval = False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, device=device)
    trainer.train(train_loader, test_loader, epochs)

if __name__ == "__main__":
    main()
