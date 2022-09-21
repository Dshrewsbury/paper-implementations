
import torch
import numpy as np


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
