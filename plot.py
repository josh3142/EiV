import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from typing import Optional, Callable, List
import PIL.Image


# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs: PIL.Image,  title: Optional[str] = None, 
    row_title: Optional[List] = None, col_title: Optional[List] = None,
    cmap: str = "viridis",
    save: Optional[str] = None, close: bool = True, **imshow_kwargs) -> None:
    
    plt.clf()
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(figsize = (10, 10), 
        nrows = num_rows, ncols = num_cols, squeeze = False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap = cmap, **imshow_kwargs)
            ax.set(xticklabels = [], yticklabels = [], xticks = [], yticks = [])

    if title is not None:
        fig.suptitle(title, fontsize = 40, y = 1)

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel = row_title[row_idx])

    if col_title is not None:
        for col_idx in range(num_cols):
            axs[0, col_idx].set(title = col_title[col_idx])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save)

    if close:
        plt.close("all")


def save_animation(idx: int, timesteps: int, samples: int, 
    transformation: Callable, interval: int = 1, cmap: str = "viridis", 
    name: str = "diffusion") -> None:
    """
    Saves an animation of the images stored in samples.
    """
    fig  = plt.figure()
    imgs = []
    for t in range(timesteps):
        if t%interval == 0:
            img = plt.imshow(transformation(samples[t][idx].cpu()), 
                cmap = cmap, animated = True)
            imgs.append([img])

    animate = animation.ArtistAnimation(fig, imgs, interval = 50, 
        blit = True, repeat_delay = 1000)
    animate.save(f'{name}.gif')


