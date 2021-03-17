import numpy
import skimage.filters
import matplotlib.pyplot as plt
from typing import List, Iterable, Optional
from matplotlib import animation
from IPython.display import HTML
import torch
import torch.nn.functional as F


def roller_2d(a: torch.tensor, b: Optional[torch.tensor] = None, radius: int = 2, shape: str = "circle"):
    """
    Performs rolling of the last two dimensions of the tensor.
    For each point consider half of the connection (the other half will be recorded by the other member of the pair).
    For example for a square shape of radius = 2 the full neighbourhood is 5x5 and for each pixel
    I only need to record 12 neighbouring pixels

    Args:
        a: First tensor to roll
        b: Second tensor to roll
        radius: size of the
        shape: Either "circle" or "square".

    Returns:
        An iterator container with the all metric of interest.
    """

    assert len(a.shape) > 2
    assert len(b.shape) > 2 or b is None

    dxdy_list = []
    for dx in range(0, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy <= 0:
                continue
            if shape == "square":
                dxdy_list.append((dx, dy))
            elif shape == "circle":
                if dx*dx + dy*dy <= radius*radius:
                    dxdy_list.append((dx, dy))
            else:
                raise Exception("Invalid shape argument. It can only be 'square' or 'circle'")

    for dxdy in dxdy_list:
        a_tmp = torch.roll(torch.roll(a, dxdy[0], dims=-2), dxdy[1], dims=-1)
        b_tmp = None if b is None else torch.roll(torch.roll(b, dxdy[0], dims=-2), dxdy[1], dims=-1)
        yield a_tmp, b_tmp


@torch.no_grad()
def compute_sparse_similarity(movie_tyx: torch.tensor,
                              index_yx: torch.tensor,
                              radius_nn: int,
                              min_threshold: float= 0.01) -> torch.sparse.FloatTensor:

    # Pad width and height with zero before rolling to avoid spurious connections due to PBC
    pad = radius_nn + 1
    pad_movie = F.pad(movie_tyx, pad=[pad, pad, pad, pad], mode="constant", value=0.0)
    pad_index = F.pad(index_yx, pad=[pad, pad, pad, pad], mode="constant", value=-1)
    row = index_yx
    row_ge_0 = (row >= 0)
    max_index = torch.max(index_yx)
    w, h = movie_tyx.shape[-2:]

    sparse_similarity = torch.sparse.FloatTensor(max_index, max_index).to(movie_tyx.device)
    # sequential loop over the allowed displacements
    for pad_movie_shifted, pad_index_shifted in roller_2d(a=pad_movie, b=pad_index, radius=radius_nn):
        v = (pad_movie * pad_movie_shifted).sum(dim=-3)[:, 0, pad:(pad + w), pad:(pad + h)]
        col = pad_index_shifted[:, 0, pad:(pad + w), pad:(pad + h)]

        # check that pixel IDs are valid and connectivity larger than threshold
        mask = (v > min_threshold) * (col >= 0) * row_ge_0

        index_tensor = torch.stack((row[mask], col[mask]), dim=0)
        tmp_similarity = torch.sparse.FloatTensor(index_tensor, v[mask],
                                                  torch.Size([max_index, max_index]))

        # Accumulate the results obtained for different displacements
        sparse_similarity.add_(tmp_similarity)

    return sparse_similarity.coalesce()






def compute_correlation_maps(ref_point_yx: numpy.ndarray, movies: numpy.ndarray):
    correlations = []
    for movie in movies:
        correlations.append(numpy.sum(movie * movie[:, ref_point_yx[1], ref_point_yx[0]].reshape(-1, 1, 1), axis=0))
    return correlations


def visualize_image_list(images: Iterable[numpy.ndarray], ncols: int = 3):
    nrows = int(numpy.ceil(float(len(images)) / ncols))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(24, 6))
    for r in range(nrows):
        for c in range(ncols):
            n = r*ncols + c
            if n < len(images):
                ax[r, c].imshow(images[n])
            else:
                ax[r, c].set_axis_off()
    plt.close(fig)
    return fig


def plot_correlation_maps(c_yx: numpy.ndarray,
                          gt_mask: numpy.ndarray,
                          correlations_maps: List[numpy.ndarray],
                          title: str):
    nrows = len(correlations_maps)
    fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(12,8))
    fig.suptitle(title)
    
    for n, corr_map in enumerate(correlations_maps):
        ax[n, 0].imshow(corr_map)
        ax[n, 0].scatter(c_yx[0], c_yx[1], color='red', s=20)
        ax[n, 1].imshow(corr_map>skimage.filters.threshold_otsu(corr_map.reshape(-1)))
        ax[n, 1].scatter(c_yx[0], c_yx[1], color='red', s=20)
        ax[n, 2].imshow(gt_mask)
        ax[n, 2].scatter(c_yx[0], c_yx[1], color='red', s=20)
        
        ax[n, 0].set_axis_off()
        ax[n, 1].set_axis_off()
        ax[n, 2].set_axis_off()
    plt.close(fig)
    return fig


def show_traces_from_movie(c_yx: numpy.ndarray, shifts: Iterable[int], movies: List[numpy.ndarray], title: str = None):
    nrows = len(movies)
    
    fig, ax = plt.subplots(nrows=nrows, figsize=(16, 8))
    fig.suptitle(title)
    for n in range(nrows):
        for shift in shifts:
            ax[n].plot(movies[n][:, c_yx[1]+shift, c_yx[0]+shift])
    plt.close(fig)
    return fig


def mu_std_moving_average(x: numpy.ndarray, 
                          axis: int, 
                          window_type: str,
                          window_size: int):
    """ The window size should be an odd integer """
    assert (window_size +1) % 2 ==0
    N = x.shape[axis]
    pad_sequence = numpy.reshape([0,0]*len(x.shape), (len(x.shape), 2))
    pad_sequence[axis, :] = window_size
    x = numpy.pad(x, pad_width=pad_sequence, mode='edge')
    
    mu_tmp = numpy.cumsum(x, axis=axis)
    mu = (numpy.roll(mu_tmp, shift=-window_size, axis=axis)-mu_tmp)/window_size
    
    f = float(window_size) / (window_size-1)
    x2_tmp = numpy.cumsum(x**2, axis=axis)
    var = (numpy.roll(x2_tmp, shift=-window_size, axis=axis) - x2_tmp)/(window_size-1) - f * mu**2
    std = numpy.sqrt(numpy.clip(var, a_min=1E-6, a_max=None))

    if window_type == 'center':
        start = (window_size -1)//2
    elif window_type == 'left':
        start = 0
    elif window_type == 'right':
        start = window_size-1
    else:
        raise Exception('window_type can only be = "center" , "left", "right"')
    
    if axis == 0 or len(x.shape)+axis == 0:
        return mu[start:start+N], std[start:start+N]
    elif axis == 1 or len(x.shape)+axis == 1:
        return mu[:, start:start+N], std[:, start:start+N]
    elif axis == 2 or len(x.shape)+axis == 2:
        return mu[:, :, start:start+N], std[:, :, start:start+N]
    elif axis == 3 or len(x.shape)+axis == 3:
        return mu[:, :, :, start:start+N], std[:, :, :, start:start+N]
    
    
def show_movie(movie: numpy.array,
               interval: int = 10,
               figsize: tuple = (16, 6)):
    assert isinstance(movie, numpy.ndarray)
    assert len(movie.shape) == 3
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("clean movie")
    im = ax.imshow(movie[0], cmap=plt.cm.Greys_r)    
    
    def animate(i):
        im.set_data(movie[i])
 
    anim = animation.FuncAnimation(fig, animate, frames=movie.shape[0], interval=interval)
    return HTML(anim.to_html5_video())
