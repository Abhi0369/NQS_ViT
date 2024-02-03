from einops import rearrange
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn




class ScaleNorm(nn.Module):
    """ScaleNorm normalization."""
    dim: int
    eps: float = 1e-5
    dtype: jnp.dtype = jnp.complex64  

    @nn.compact
    def __call__(self, x):
        """Applies ScaleNorm normalization.

        Args:
            x: Input tensor.

        Returns:
            The normalized tensor.
        """
        g = self.param('g', nn.initializers.ones, (1,), self.dtype) * (self.dim ** -0.5)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scaled_x = x / jnp.maximum(norm, self.eps) * g
        return scaled_x

class RelativePositionEmbedding(nn.Module):
    """Embedding for encoding relative positions."""
    hidden_size: int
    max_relative_position: int
    embedding_init: nn.initializers.Initializer = nn.initializers.normal(dtype=jnp.complex64)
    # embedding_init: nn.initializers.Initializer = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform', dtype=jnp.complex64)


    def setup(self):
        """Initializes the embedding table for relative positions."""

        self.relative_pos_embed = self.param('relative_positions_embeddings',
                                             self.embedding_init,
                                             (2 * self.max_relative_position + 1, self.hidden_size))

    def __call__(self, length):
        """Generates a tensor of size (length x length) filled with relative positional encodings.

        Args:
            length: The length of the sequence (number of patches).

        Returns:
            A tensor with shape [1, length, hidden_size] containing relative positional embeddings.
        """
        range_vec = jnp.arange(length)
        range_mat = range_vec[:, None] - range_vec[None, :]
        range_mat_clipped = jnp.clip(range_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = range_mat_clipped + self.max_relative_position
        embeddings = self.relative_pos_embed[final_mat]
        embeddings = embeddings.mean(0,keepdims=True)
        return embeddings

class PatchEmbeddings(nn.Module):
    """Embeddings for the patches extracted from an input configuration."""
    hidden_size: int
    patch_size: int

    @nn.compact
    def __call__(self, x):
        """Applies patch embeddings to input images.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels].

        Returns:
            A tensor of shape [batch_size, num_patches, hidden_size] with embedded patches.
        """

        p = self.patch_size
        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = nn.Dense(features=self.hidden_size, param_dtype=jnp.complex64)(x)

        return x
    


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    hidden_size: int
    num_heads: int
    qkv_bias: bool

    @nn.compact
    def __call__(self, x, mask=None):
        """Applies multi-head self-attention to the input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size].
            mask: (Optional) Attention mask.
            deterministic: (Optional) If True, does not apply dropout.

        Returns:
            The output tensor after applying multi-head self-attention.
        """
        attention_head_size = self.hidden_size // self.num_heads
        all_head_size = self.num_heads * attention_head_size

        query = nn.Dense(features=all_head_size, param_dtype=jnp.complex64, use_bias=self.qkv_bias)(x)
        key = nn.Dense(features=all_head_size, param_dtype=jnp.complex64, use_bias=self.qkv_bias)(x)
        value = nn.Dense(features=all_head_size, param_dtype=jnp.complex64, use_bias=self.qkv_bias)(x)

        def split_heads(x):
            """Splits the last dimension into (num_heads, attention_head_size)."""
            return x.reshape(x.shape[:-1] + (self.num_heads, attention_head_size)).transpose(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        attention_scores = jnp.einsum('bhqd, bhkd->bhqk', query, key)

        attention_scores = attention_scores / jnp.sqrt(attention_head_size)

        if mask is not None:
            attention_scores = jnp.where(mask, attention_scores, jnp.full_like(attention_scores, -1e9))

        attention_weights = nn.softmax(attention_scores, axis=-1)
        attention_output = jnp.einsum('bhqk, bhvd->bhqd', attention_weights, value)
 
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(x.shape[:-1] + (all_head_size,))


        return attention_output

class EncoderBlock(nn.Module):
    """A single block of the Transformer encoder."""
    num_heads: int
    hidden_size: int
    use_scale_norm: bool = False  # Determines whether to use ScaleNorm.

    @nn.compact
    def __call__(self, x):
        """Applies the encoder block operations on the input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size].

        Returns:
            The output tensor after processing with the encoder block with the same shape.
        """
        attn = MultiHeadAttention(num_heads=self.num_heads, hidden_size=self.hidden_size, qkv_bias=True)(x)
        x = x + attn

     
        if self.use_scale_norm:
            x = ScaleNorm(dim=self.hidden_size, dtype=jnp.complex64)(x)
        else:
            x = nn.LayerNorm(param_dtype=jnp.complex64)(x)

        # MLP
        x = nn.Dense(features=self.hidden_size, param_dtype=jnp.complex64)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_size, param_dtype=jnp.complex64)(x)

        if self.use_scale_norm:
            x = ScaleNorm(dim=self.hidden_size, dtype=jnp.complex64)(x)
        else:
            x = nn.LayerNorm(param_dtype=jnp.complex64)(x)

        return x


    
class VisionTransformer(nn.Module):
    """Vision Transformer model."""
    patch_size: int
    hidden_size: int
    lattice_size: int
    num_heads: int
    num_layers: int
    num_classes: int
    num_channels: int
    use_cls_token: bool = False
    use_relative_pos_embedding: bool = False
    use_scale_norm: bool = False  # Determines the normalization method in the encoder blocks.

    @nn.compact
    def __call__(self, x):
        """Applies the Vision Transformer model to an input batch of configurations.

        Args:
            x: Input tensor of shape [batch_size, height, width, channels].

        Returns:
            log psi of shape [batch_size].
        """
        x = PatchEmbeddings(patch_size=self.patch_size, hidden_size=self.hidden_size)(x)

        # Conditional inclusion of a class token
        if self.use_cls_token:
            cls_token = self.param('cls_token', nn.initializers.normal(dtype=jnp.complex64), (1, 1, self.hidden_size))
            cls_token = jnp.tile(cls_token, [x.shape[0], 1, 1])
            x = jnp.concatenate([cls_token, x], axis=1)

        # Apply either relative position embedding or a learnable random position embedding
        if self.use_relative_pos_embedding:
            length = x.shape[1]

            rel_pos_embedding = RelativePositionEmbedding(self.hidden_size, max_relative_position=2)(length)
            x = x + rel_pos_embedding
        else:
            length = x.shape[1]
            # embedding_init = jax.nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform', dtype=jnp.float32)
            embedding_init= nn.initializers.normal(dtype=jnp.complex64)
            pos_embedding = self.param('pos_embedding', embedding_init, (length, self.hidden_size))
            x = x + pos_embedding
   

        # Process through a series of Transformer encoder blocks
        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads, hidden_size=self.hidden_size, use_scale_norm=self.use_scale_norm)(x)

        x = complex_log_cosh(x)

        if self.use_cls_token:
            x = x[:, 0, :]  # Use the class token
        else:
            x = jnp.mean(x, axis=(1))  # Average pooling

        x = nn.Dense(features=self.num_classes, param_dtype=jnp.complex64)(x)

        return x.squeeze()


def complex_log_cosh(z):
    cosh_z = (jnp.exp(z) + jnp.exp(-z)) / 2
    return jnp.log(cosh_z)