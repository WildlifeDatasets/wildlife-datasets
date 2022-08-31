import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def prepare_batch(batch, device='cpu'):
    if isinstance(batch, list):
        x, y = batch
    else:
        raise NotImplementedError('Batch needs to be (x, y) tuple.')

    if isinstance(x, list):
        x = [tensor.to(device) for tensor in x]
    else:
        x = x.to(device)

    y = y.to(device)
    return x, y

def grid_plot(img_batch, save_as=None, nrows=6, already_grid=False, figsize=None):
    '''
    Plot tensor batch of images. Input dimension format: (B,C,W,H)
    '''
    if already_grid:
        grid = img_batch
    else:
        grid = make_grid(img_batch, nrow=nrows, normalize=True)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))
    if figsize == None:
        figsize = (nrows, nrows)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(grid)
    if save_as is not None:
        fig.savefig(save_as)
        plt.close(fig)
    else:
        plt.show()