import numpy as np
import torch
from specs import *
from boltons.cacheutils import cachedproperty
from typing import Dict


class CameraSimulator:
    KERNEL_REPEATS = 5
    
    def __init__(self, cam_specs: CameraSpecs):
        self.cam_specs = cam_specs
        self.rng = np.random.default_rng()
    
    @cachedproperty
    def _wx(self):
        return int(np.ceil(CameraSimulator.KERNEL_REPEATS * self.cam_specs.psf_lengthscale))

    @cachedproperty
    def _wy(self):
        return int(np.ceil(CameraSimulator.KERNEL_REPEATS * self.cam_specs.psf_lengthscale))
    
    @cachedproperty
    def gaussian_psf_kernel(self) -> torch.Tensor:
        wx = self._wx
        wy = self._wy
        x = torch.arange(-wx, wx + 1, dtype=torch.float64)[None, :]
        y = torch.arange(-wy, wy + 1, dtype=torch.float64)[:, None]
        struct = torch.exp(
            - 0.5 * x.pow(2) / (self.cam_specs.psf_lengthscale ** 2)
            - 0.5 * y.pow(2) / (self.cam_specs.psf_lengthscale ** 2))
        struct = struct / struct.sum()
        kern = struct[None, None, ...]
        return kern

    def __call__(self, fluorophore_count_yx: np.ndarray) -> Dict:
        fluorophore_count_yx = np.pad(
            fluorophore_count_yx,
            pad_width=((self._wy, self._wy), (self._wx, self._wx)),
            mode='reflect')
        photon_rate_yx_torch = torch.tensor(
            self.cam_specs.photon_per_fluorophore * fluorophore_count_yx, dtype=torch.float64)
        blurred_photon_rate_yx = torch.nn.functional.conv2d(
            photon_rate_yx_torch[None, None, ...],
            self.gaussian_psf_kernel)[0, 0, ...].cpu().numpy()
        noisy_readout_yx = (
            self.cam_specs.readout_per_photon * self.rng.poisson(blurred_photon_rate_yx) +
            self.cam_specs.gaussian_noise_std * self.rng.normal(size=blurred_photon_rate_yx.shape) +
            self.cam_specs.dc_offset)
        noisy_readout_yx[noisy_readout_yx < 0] = 0
        clean_readout_yx = self.cam_specs.readout_per_photon * blurred_photon_rate_yx + self.cam_specs.dc_offset
        return {
            'noisy_readout_yx': noisy_readout_yx,
            'clean_readout_yx': clean_readout_yx
        }
