import torch.nn as nn
import torch
from typing import Optional, Tuple

class SiglipVisionConfig:
  def __init__(
      self,
      hidden_size=768,
      intermediate_size = 3072,
      num_hidden_layers=12,
      num_attention_heads=12,
      num_channels = 3,
      image_size = 224,
      patch_size = 16,
      layer_norm_eps=1e-6,
      attention_dropout=0.0,
      num_image_tokens: int = None,
      **kwargs
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.layer_norm_eps = layer_norm_eps
    self.attention_dropout = attention_dropout
    self.num_image_tokens = num_image_tokens


class PatchEmbeddings(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.image_size = config.image_size
    self.hidden_size = config.hidden_size
    self.num_channels = config.num_channels
    self.patch_size = config.patch_size

    self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x):
    x = self.projection(x)
    return x

class SigLipVisionEmbeddings(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.image_size = config.image_size
    self.hidden_size = config.hidden_size
    self.num_channels = config.num_channels
    self.patch_size = config.patch_size
    self.num_patches = (self.num_channels // self.patch_size) ** 2
    self.patch_embeddings = PatchEmbeddings(config)
    self.num_positions = self.num_patches
    self.position_embeddings = nn.Embedding(self.num_positions, self.hidden_size)

  def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    _, _, height, width = pixel_values.shape
    patch_embeds = self.patch_embeddings(pixel_values)
    #[batch_size, hidden_size, num_patches]
    embeddings = patch_embeds.flatten(2)
    #[batch_size, hidden_size, num_patches] -> [batch_size, num_patches, hidden_size]
    embeddings = embeddings.transpose(1, 2)
    #[batch_size, num_patches, hidden_size]
    embeddings = self.position_embeddings(patch_embeds)
    #[batch_size, num_oatches, hidden_size]

    return embeddings
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

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Extract parameters from config
        self.d_model = config.hidden_size
        assert d_model % n_heads == 0, "d_model must be divisible by h"
        self.d_k = self.d_model // config.num_attention_heads
        self.n_heads = config.num_attention_heads

        # Initialize projection layers
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model)

        # Initialize attention mechanism (assumed to be defined elsewhere)
        self.attention = ScaledDotProduct()
        self.dropout = config.attention_dropout
        # Initialize dropout layers
        self.attention_dropout = nn.Dropout(self.dropout)

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

        # Return outputs
        # return (output, attention_scores if output_attentions else None)
        #if output_attentions:
        return output, attention_scores  # Return both output and attention scores
        #else:
            #return output, None  # Return None for attention scores
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.attention = SiglipAttention(config)
    self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
    self.mlp = SiglipMLP(config)
    self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

      residual = hidden_states
      hidden_states = self.layer_norm1(hidden_states)
      hidden_states, _ = self.attention(hidden_states)
      hidden_states = residual + hidden_states
      residual = hidden_states
      hidden_states = self.layer_norm2(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states

      return hidden_states

class SiglipEncoder(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
      hidden_states = input_embeds

      for encoder_layer in self.layers:
        hidden_states = encoder_layer(hidden_states)

      return hidden_states

class SiglipVisionTransformer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.embeddings = SiglipVisionEmbeddings(config)
    self.encoder = SiglipEncoder(config)
    self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    hidden_states = self.embeddings(pixel_values)
    last_hidden_states = self.encoder(hidden_states)
    last_hidden_states = self.post_layernorm(last_hidden_states)
    return last_hidden_states

class SiglipVisionModel(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.vision_model = SiglipVisionTransformer(config)

  def forward(self, pixel_values: torch.Tensor) -> Tuple:
    return self.vision_model(pixel_values)
