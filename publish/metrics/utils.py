from torchtyping import TensorType
import numpy as np
import matplotlib.pyplot as plt
import torch

def heatmap(data, vmin=None, vmax=None, cmap='coolwarm'):
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(img)
    plt.close(fig)
    return fig
