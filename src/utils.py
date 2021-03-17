import os
import numpy as np
import torch
from specs import *
from allen_data import \
    ProcessedAllenNeuronMorphology, \
    ProcessedAllenNeuronElectrophysiology


def load_morphology(
        optosynth_data_path: str,
        cell_id: int) -> ProcessedAllenNeuronMorphology:
    return ProcessedAllenNeuronMorphology.from_file(
        os.path.join(optosynth_data_path, 'processed_morphology', f'{cell_id}_processed_morphology.pkl'))


def load_electrophysiology(
        optosynth_data_path: str,
        cell_id: int) -> ProcessedAllenNeuronElectrophysiology:
    return ProcessedAllenNeuronElectrophysiology.from_file(
        os.path.join(optosynth_data_path, 'processed_electrophysiology', f'{cell_id}_processed_electrophysiology.pkl'))


def scale_array(arr: np.ndarray, min_value: float, max_value: float, eps: float = 1e-9) -> np.ndarray:
    _min_value, _max_value = arr.min(), arr.max()
    return min_value + (max_value - min_value) * (arr - _min_value) / (_max_value - _min_value + eps)


def generate_correlated_noise_1d(
        min_value: float,
        max_value: float,
        length: int,
        smoothing_radius: float,
        smoothing_repeats: int,
        device: torch.device,
        dtype: torch.dtype) -> np.ndarray:
    """Samples from a 1D Gaussian process with Gaussian kernel."""
    
    # smoothing kernel
    w = int(np.ceil(smoothing_radius * smoothing_repeats))
    x = torch.arange(-w, w + 1, dtype=dtype, device=device)
    struct = torch.exp(- 0.5 * x.pow(2) / (smoothing_radius ** 2))
    struct = struct / struct.sum()
    kern = struct[None, None, ...]

    # uncorrelated noise
    noise_x = torch.rand((length + 2 * w,), dtype=dtype, device=device)

    # calculate locally-summed weight
    smooth_noise_x = torch.nn.functional.conv1d(noise_x[None, None, ...], kern)[0, 0, ...]

    return scale_array(smooth_noise_x, min_value, max_value)


def generate_correlated_noise_2d(
        min_value: float,
        max_value: float,
        width: int,
        height: int,
        smoothing_radius_x: float,
        smoothing_radius_y: float,
        smoothing_repeats: int,
        device: torch.device,
        dtype: torch.dtype) -> np.ndarray:
    """Samples from a 2D Gaussian process with Gaussian kernel."""
    
    # smoothing kernel
    wx = int(np.ceil(smoothing_radius_x * smoothing_repeats))
    wy = int(np.ceil(smoothing_radius_y * smoothing_repeats))
    x = torch.arange(-wx, wx + 1, dtype=dtype, device=device)[:, None]
    y = torch.arange(-wy, wy + 1, dtype=dtype, device=device)[None, :]
    struct = torch.exp(
        - 0.5 * x.pow(2) / (smoothing_radius_x ** 2)
        - 0.5 * y.pow(2) / (smoothing_radius_y ** 2))
    struct = struct / struct.sum()
    kern = struct[None, None, ...]

    # uncorrelated noise
    noise_xy = torch.rand(
        (width + 2 * wx, height + 2 * wy), dtype=dtype, device=device)

    # calculate locally-summed weight
    smooth_noise_xy = torch.nn.functional.conv2d(noise_xy[None, None, ...], kern)[0, 0, ...]

    return scale_array(smooth_noise_xy, min_value, max_value).T


def paste_image(
        source_yx: np.ndarray,
        target_yx: np.ndarray,
        source_anchor_x: int,
        source_anchor_y: int,
        target_anchor_x: int,
        target_anchor_y: int) -> np.ndarray:
    """Pastes a source image (ndarray) into a target image (ndarray) according to
    the given anchoring points."""
    dx = target_anchor_x - source_anchor_x
    dy = target_anchor_y - source_anchor_y
    x0t, y0t = dx, dy
    x1t, y1t = dx + source_yx.shape[1], dy + source_yx.shape[0]
    x0s, y0s = 0, 0
    x1s, y1s = source_yx.shape[1], source_yx.shape[0]
    if x0t < 0:
        x0s -= x0t
        x0t = 0
    if y0t < 0:
        y0s -= y0t
        y0t = 0
    if x1t > target_yx.shape[1]:
        x1s -= (x1t - target_yx.shape[1])
        x1t = target_yx.shape[1]
    if y1t > target_yx.shape[0]:
        y1s -= (y1t - target_yx.shape[0])
        y1t = target_yx.shape[0] 
    target_yx[y0t:y1t, x0t:x1t] += source_yx[y0s:y1s, x0s:x1s]
