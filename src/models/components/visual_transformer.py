import torch
from torch import nn
import numpy as np
# from transformer_utils import get_positional_embeddings
# from transformer_utils import divide_into_patches


class ViT(nn.Module):
    def __init__(self,
                 channels: int = 1,
                 height: int = 28,
                 width: int = 28,
                 n_patches: int = 7,
                 n_blocks: int = 2,
                 hidden_d: int = 8,
                 n_heads: int = 2,
                 out_d: int = 10
                 ):
        super(ViT, self).__init__()

        # Attributes
        self.channels = channels
        self.height = height
        self.width = width
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert self.height % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert self.width % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (self.height / n_patches, self.width / n_patches)

        # 1) Linear mapper
        self.input_d = int(self.channels * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token, captures info about other tokens, use this to classify image
        # If we want to do another downstream task(ex: classifying a digit as higher than 5 or lower)
        # just add another token and a classifier that takes as input this new token
        # Note: classification token is first token of each sequence
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        # is a function of the number of elements in the sequence and dimensionality of each
        # element thus always is a 2d tensor
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d).clone())
        self.pos_embed.requires_grad = False

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = divide_into_patches(images, self.n_patches).to(self.pos_embed.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)  # Map to output dimension, output category distribution


class SelfAttention(nn.Module):
    """
    For a single image, each patch is updated based on some similarity measure with other patches\
    Allows modeling the relationship between inputs
    Note: Using loops is not the most efficient way for computing attention, but it's clearer for learning
    """
    def __init__(self, d, n_heads=2):
        super(SelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.multihead_attention = SelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # Residual Connections
        out = x + self.multihead_attention(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


def divide_into_patches(images, n_patches):
    """
    Function  for Visual Transformer that divides an image into patches and maps to a subvector

    Input has size (N,Channels,Height,Width), for base example we are using MNIST so a single channel
    MNIST images are 28x28, so we will reshape input (N, 1, 28, 28) to: (N, PxP, HxC/P x WxC/P)
    = (N, 7x7, 4x4) = (N, 49, 16)
    P = # of patches

    Note: This is a very inefficient way of doing this, but it's more intuitive for learning, which is the point of this
    visual transformer
    """
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


if __name__ == "__main__":
    _ = ViT()
