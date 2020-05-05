import numpy as np
import torch
from specs import *
from utils import \
    generate_correlated_noise_1d, \
    generate_correlated_noise_2d


class BackgroundFluorescenceGenerator:
    def __init__(
            self,
            bg_specs: BackgroundFluorescenceSpecs,
            opto_specs: OptosynthSpecs,
            device: torch.device,
            dtype: torch.dtype):
        
        self.bg_specs = bg_specs
        self.opto_specs = opto_specs
        self.device = device
        self.dtype = dtype
        
        # generate dynamic backgrounds
        if bg_specs.dynamic_n_components > 0:
            dynamic_bg_nyx = np.zeros((bg_specs.dynamic_n_components, opto_specs.height, opto_specs.width))
            for i_bg in range(bg_specs.dynamic_n_components):
                dynamic_bg_yx = generate_correlated_noise_2d(
                    min_value=-1.0 / bg_specs.dynamic_n_components,
                    max_value=+1.0 / bg_specs.dynamic_n_components,
                    width=opto_specs.width,
                    height=opto_specs.height,
                    smoothing_radius_x=bg_specs.dynamic_x_lengthscale * opto_specs.scale_factor,
                    smoothing_radius_y=bg_specs.dynamic_y_lengthscale * opto_specs.scale_factor,
                    smoothing_repeats=5,
                    device=device,
                    dtype=dtype).cpu().numpy()
                dynamic_bg_yx = dynamic_bg_yx - np.mean(dynamic_bg_yx)
                dynamic_bg_nyx[i_bg, ...] = dynamic_bg_yx    
            self.dynamic_bg_nyx = dynamic_bg_nyx
        else:
            self.dynamic_bg_nyx = None
            
        # generate the static background
        self.static_bg_yx = generate_correlated_noise_2d(
            min_value=bg_specs.static_min_total_fluorophore_density / opto_specs.scale_factor**2,
            max_value=bg_specs.static_max_total_fluorophore_density / opto_specs.scale_factor**2,
            width=opto_specs.width,
            height=opto_specs.height,
            smoothing_radius_x=bg_specs.static_x_lengthscale * opto_specs.scale_factor,
            smoothing_radius_y=bg_specs.static_y_lengthscale * opto_specs.scale_factor,
            smoothing_repeats=5,
            device=device,
            dtype=dtype).cpu().numpy()
        
    def generate(self, n_frames: int, sampling_rate: float) -> np.ndarray:
        assert n_frames > 0
        assert sampling_rate > 0
        
        # generate dynamic background
        if self.dynamic_bg_nyx is not None:
            activation_nt = np.zeros((self.bg_specs.dynamic_n_components, n_frames))
            for i_bg in range(self.bg_specs.dynamic_n_components):
                activation_t = generate_correlated_noise_1d(
                    min_value=-self.bg_specs.dynamic_fluorophore_density_scale / self.opto_specs.scale_factor**2,
                    max_value=self.bg_specs.dynamic_fluorophore_density_scale / self.opto_specs.scale_factor**2,
                    length=n_frames,
                    smoothing_radius=(sampling_rate / self.bg_specs.dynamic_temporal_frequency),
                    smoothing_repeats=5,
                    device=self.device,
                    dtype=self.dtype).cpu().numpy()
                activation_nt[i_bg, :] = activation_t
            lowest_value = self.bg_specs.dynamic_fluorophore_density_scale / self.opto_specs.scale_factor**2
            dynamic_bg_tyx = lowest_value + np.einsum(
                'nt,nyx->tyx', activation_nt, self.dynamic_bg_nyx)
        else:
            dynamic_bg_tyx = no.zeros(
                (n_frames, self.opto_specs.height, self.opto_specs.width))
        
        return self.static_bg_yx[None, ...] + dynamic_bg_tyx
