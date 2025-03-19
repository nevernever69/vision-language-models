import torch
import numpy as np
import pandas as pd
import math
from torch import nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class ScaledDotProduct(nn.Module):
  def __init__(self):
    super(ScaledDotProduct, self).__init__()
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, q, k, v, mask, dropout: nn.Dropout):
    d_k = q.shape[-1]
    # Fixed the order of operations
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled.masked_fill_(mask == 0, -1e9)

    attention = self.softmax(scaled)
    if dropout is not None:
        attention = dropout(attention)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, drop_prob = 0.0, hidden_drop_prob = 0.0):
        super().__init__()
        # Extract parameters from config
        self.d_model = d_model
        assert d_model % n_heads == 0, "d_model must be divisible by h"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        bias = True

        # Initialize projection layers
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=bias)

        # Initialize attention mechanism (assumed to be defined elsewhere)
        self.attention = ScaledDotProduct()

        # Initialize dropout layers
        self.attention_dropout = nn.Dropout(drop_prob)
        self.output_dropout = nn.Dropout(hidden_drop_prob)

    def forward(self, x, mask=None,output_attentions=False):
        # Self-attention: q = k = v = x

        q = k = v = x
        #mask = None  # No mask, as per ViT implementation

        # Project inputs
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Reshape and transpose for multi-head attention
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention
        x, attention_scores = self.attention(query, key, value, mask, self.attention_dropout)

        # Reshape output
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        # Apply output projection and dropout
        output = self.w_o(x)
        output = self.output_dropout(output)

        # Return outputs
        # return (output, attention_scores if output_attentions else None)
        if output_attentions:
            return output, attention_scores  # Return both output and attention scores
        else:
            return output, None  # Return None for attention scores

class PostionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()
    self.d_model = d_model
    self.max_seq_length = max_seq_length

    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0)/d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    seq_len = x.size(1)
    return x + self.pe[:, :seq_len]

class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, d_model, mlp_ratio=4, hidden_drop_prob = 0.0):
        super().__init__()
        self.intermediate_size = mlp_ratio * d_model
        self.dense_1 = nn.Linear(d_model, self.intermediate_size)
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(self.intermediate_size, d_model)
        self.dropout = nn.Dropout(hidden_drop_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.image_size = img_size
        self.patch_size = patch_size
        self.num_channels = n_channels
        self.hidden_size = d_model
        self.num_patches = (img_size[0] * img_size[1] ) // (patch_size[0] * patch_size[1])
        # Calculate the number of patches from the image size and patch size
        #self.num_patches = (self.image_size // self.patch_size) ** 2
        print(f"Number of patches: {self.num_patches}, Expected tokens with CLS: {self.num_patches + 1}")

        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, d_model, img_size, patch_size, n_channels, hidden_drop_prob=0.0):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings(d_model, img_size, patch_size, n_channels)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, d_model))
        self.dropout = nn.Dropout(hidden_drop_prob)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)

        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # if output_attentions:
        #     return (x, attention_probs)
        # else:
        #     return (x, None)  # Return None for attention scores when not requested
        return x
class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        # Create a list of transformer blocks
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            block = Block(d_model, n_heads)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x = block(x, output_attentions=output_attentions)
            # if output_attentions:
            #     all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        # if not output_attentions:
        #     return (x, None)
        # else:
        #     return (x, all_attentions)
        return x
class VisionEncoder(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self,d_model,img_size,patch_size, n_channels, n_heads,n_layers, emb_dim):
        super().__init__()

        self.image_size = img_size
        self.emb_dim = emb_dim
        self.hidden_size = d_model
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.img_size = img_size
        # Create the embedding module
        self.embedding = Embeddings(d_model, img_size, patch_size, n_channels)
        # Create the transformer encoder module
        self.encoder = Encoder(d_model, n_heads, n_layers)
        self.ln_final = nn.LayerNorm(d_model)

        # Create a linear layer to project the encoder's output to the number of classes
        self.projection = nn.Parameter(torch.randn(self.hidden_size, self.emb_dim))
        # Initialize the weights


    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_outputs = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        encoder_output = self.ln_final(encoder_outputs)
        pooled_output = encoder_output[:, 0]

        #logits = encoder_output[:, 0]
        # Return the logits and the attention probabilities (optional)
        if self.projection is not None:
            pooled_output = pooled_output @ self.projection

        # Normalize for contrastive learning
        pooled_output = pooled_output / torch.norm(pooled_output, dim=-1, keepdim=True)

        # if not output_attentions:
        return pooled_output
        # else:
        #     return pooled_output, all_attentions


def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        # Adding SOT and EOT tokens
        out = chr(2) + text + chr(3)

        # Truncate if length exceeds max_seq_length
        if len(out) > max_seq_length:
            out = out[:max_seq_length]

        # Add padding if needed
        out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])

        # Encode the text
        out = torch.IntTensor(list(out.encode("utf-8")))

        # Create the mask
        mask = torch.ones(len(out.nonzero()))

        # Pad the mask to max_seq_length
        if len(mask) < max_seq_length:
            mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
        else:
            mask = mask.type(torch.IntTensor)
    else:
        # Decode the text
        out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
        out = "".join(out)
        mask = None

    return out, mask

class TextEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
      super().__init__()

      self.max_seq_length = max_seq_length

      self.embed = nn.Embedding(vocab_size, d_model)

      self.positional_embedding = PostionalEncoding(d_model, max_seq_length)

      self.transformer_encoder = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])

      self.projection = nn.Parameter(torch.randn(d_model, emb_dim))

   # For training
  def forward(self, text, mask = None):

      x = self.embed(text)

      x = self.positional_embedding(x)

      for encoder_layer in self.transformer_encoder:
          x = encoder_layer(x)

      #The output of the encoder layers is the text features. We are going to be using the features from the EOT embedding.

      x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:,0],dim=1),1)]

      if self.projection is not None:
         x = x @ self.projection

      x = x / torch.norm(x, dim=-1, keepdim = True)

      return x

class TextEncoder_Retrieval(nn.Module):
  def __init__(self, vocab_size, d_model, max_seq_length, n_layers,n_heads, emb_dim):
      super().__init__()

      self.max_seq_length = max_seq_length

      self.embed = nn.Embedding(vocab_size, d_model)

      self.positional_embedding = PostionalEncoding(d_model, max_seq_length)

      self.transformer_encoder = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])

      self.projection = nn.Parameter(torch.randn(d_model, emb_dim))


 # # For image retrieval
  def forward(self, text, mask=None):
        x = self.embed(text)
        x = self.positional_embedding(x)

        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)

        if mask is not None:
            # Get the lengths of each sequence (i.e., find the last non-padded token)
            seq_lengths = mask.sum(dim=1) - 1  # Subtract 1 to get the index
            x = x[torch.arange(text.shape[0]), seq_lengths]
        else:
            x = x[:, -1]  # If no mask is provided, take the last token in the sequence.

        if self.projection is not None:
            x = x @ self.projection

        x = x / torch.norm(x, dim=-1, keepdim=True)

        return x

class CLIP(nn.Module):

    def __init__(self, emb_dim, vit_layers, vit_d_model, img_size, patch_size, n_channels, vit_heads, vocab_size, max_seq_length, text_heads, text_layers, text_d_model, retrieval = False):
        super().__init__()

        self.vision_encoder = VisionEncoder(vit_d_model, img_size, patch_size, n_channels, vit_heads, vit_layers, emb_dim) #Vision Encoder

        self.text_encoder = TextEncoder(vocab_size, text_d_model, max_seq_length, text_layers, text_heads, emb_dim) #Text Encoder

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def CLIPLoss(self, logits, device = "cuda"):
        #Symmetric or Contrastive loss
    # arange generates a list between 0 and n-1
        labels = torch.arange(logits.shape[0]).to(device)  # For row 1 we want 1,1 to be max, and row n-1 we want (n-1,n-1) text pairs to be max --> time 15.43 umar

        loss_v = nn.functional.cross_entropy(logits.transpose(-2,-1), labels)

        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = (loss_v + loss_t) / 2

        return loss

    def forward(self, image, text, mask=None):
      V_e = self.vision_encoder(image)  # Extract just the embeddings
      T_e = self.text_encoder(text, mask)  # Extract just the embeddings

      logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)

      loss = self.CLIPLoss(logits, self.device)

      return loss

