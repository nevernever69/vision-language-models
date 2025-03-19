import torch
from PIL import Image
import torchvision.transforms as transforms
from Model import CLIP, tokenizer  # Import your CLIP model and tokenizer

# Hyperparameters (should match training)
emb_dim = 128
vit_d_model = 32
img_size = (80, 80)      # Use the image size from training
patch_size = (5, 5)
n_channels = 3
vit_layers = 8
vit_heads = 4

vocab_size = 256
text_d_model = 64
max_seq_length = 128
text_heads = 8
text_layers = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load your trained CLIP model
model = CLIP(emb_dim, vit_layers, vit_d_model, img_size, patch_size,
             n_channels, vit_heads, vocab_size, max_seq_length,
             text_heads, text_layers, text_d_model, retrieval=False).to(device)
model.load_state_dict(torch.load("clip.pt", map_location=device))
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

# List of fashion captions/classes (update with your dataset labels)
fashion_captions = [
    "t-shirt", "shirt", "dress", "jeans", "skirt", "jacket", "sneakers", "bag",
    "lipstick", "eyelash", "nail polish"
]

# Generate text embeddings for each caption
text_embeddings = []
for caption in fashion_captions:
    # Tokenize the caption
    tokens, mask = tokenizer(caption)
    # Convert to tensor and add batch dimension if needed
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    mask = torch.tensor(mask).unsqueeze(0).to(device)
    # Compute text embedding using the text encoder
    with torch.no_grad():
        embedding = model.text_encoder(tokens, mask=mask)
    text_embeddings.append(embedding.squeeze(0))
# Stack embeddings into one tensor: (num_classes, emb_dim)
text_embeddings = torch.stack(text_embeddings, dim=0)
# Normalize text embeddings
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Load and preprocess the image you want to detect
image_path = "Final.jpg"  # update with your image path
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Compute the image embedding
with torch.no_grad():
    image_embedding = model.vision_encoder(image)
# Normalize the image embedding
image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

# Compute cosine similarities between the image and each text embedding
# Since embeddings are normalized, a dot product equals the cosine similarity
similarities = (image_embedding @ text_embeddings.T).squeeze(0)

# Get the index of the highest similarity score
predicted_idx = similarities.argmax().item()
predicted_caption = fashion_captions[predicted_idx]

print("Detection Result:")
print(f"Predicted class: {predicted_caption}")
print(f"Similarity scores: {similarities.cpu().numpy()}")

