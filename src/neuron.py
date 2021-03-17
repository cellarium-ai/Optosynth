import numpy as np
import torch
from specs import *
from PIL import Image
from boltons.cacheutils import cachedmethod
from scipy.interpolate import InterpolatedUnivariateSpline
from allen_data import ProcessedAllenNeuronMorphology, ProcessedAllenNeuronElectrophysiology
from utils import generate_correlated_noise_2d
from v2f import VoltageToFluorescenceConverter


class SyntheticNeuronFluorescenceGenerator:
    def __init__(
            self,
            morph: ProcessedAllenNeuronMorphology,
            ephys: ProcessedAllenNeuronElectrophysiology,
            opto_specs: OptosynthSpecs,
            neuron_specs: SyntheticNeuronSpecs,
            v2f_specs: VoltageToFluorescenceSpecs,
            device: torch.device,
            dtype: torch.dtype):
        
        self.morph = morph
        self.ephys = ephys
        self.opto_specs = opto_specs
        self.neuron_specs = neuron_specs
        self.v2f_specs = v2f_specs
        self.device = device
        self.dtype = dtype
        
        # make a PIL image
        img = Image.fromarray(morph.mask).convert("RGB")
        
        # random rotation angle
        angle = np.random.rand() * 360
        
        # rotate soma center position and mask
        xs = morph.soma_coords[0] - img.size[0] / 2
        ys = morph.soma_coords[1] - img.size[1] / 2
        new_xs = np.cos(angle * np.pi / 180) * xs + np.sin(angle * np.pi / 180) * ys
        new_ys = - np.sin(angle * np.pi / 180) * xs + np.cos(angle * np.pi / 180) * ys        
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        xs = img.size[0] / 2 + new_xs
        ys = img.size[1] / 2 + new_ys
        
        # resize
        old_width, old_height = img.size
        width = int(opto_specs.scale_factor * old_width / morph.global_scale_factor)
        height = int(opto_specs.scale_factor * old_height / morph.global_scale_factor)
        img = img.resize((width, height), resample=Image.BICUBIC)
        xs = xs * opto_specs.scale_factor / morph.global_scale_factor
        ys = ys * opto_specs.scale_factor / morph.global_scale_factor
        
        # mask
        mask = np.asarray(img).sum(-1) > 0
        
        # random reporter density
        reporter_density = generate_correlated_noise_2d(
            min_value=neuron_specs.min_reporter_density / opto_specs.scale_factor**2,
            max_value=neuron_specs.max_reporter_density / opto_specs.scale_factor**2,
            width=width,
            height=height,
            smoothing_radius_x=neuron_specs.reporter_density_var_lengthscale * opto_specs.scale_factor,
            smoothing_radius_y=neuron_specs.reporter_density_var_lengthscale * opto_specs.scale_factor,
            smoothing_repeats=5,
            device=device,
            dtype=dtype).cpu().numpy()
        
        # fluorescence intensity
        reporter_mask = np.asarray(img.convert('F'))
        reporter_mask = reporter_mask / np.max(reporter_mask)
        reporter_density_on_neuron = reporter_density * reporter_mask
        
        # signal propagation delay mask
        x = np.arange(0, width)[:, None]
        y = np.arange(0, height)[None, :]
        effective_velocity = neuron_specs.dendritic_backprop_velocity * opto_specs.scale_factor
        dist_from_soma = np.sqrt((x - xs) ** 2 + (y - ys) ** 2).T
        delay_mask = dist_from_soma / effective_velocity
        
        # signal attenuation mask
        effective_attenuation_lengthscale = \
            opto_specs.scale_factor * neuron_specs.dendritic_backprop_decay_lengthscale
        attenuation_mask = np.exp(- dist_from_soma / effective_attenuation_lengthscale)
        
        self.mask = mask
        self.width = width
        self.height = height
        self.density = reporter_density_on_neuron
        self.delay_mask = delay_mask
        self.attenuation_mask = attenuation_mask
        self.xs = xs
        self.ys = ys
        
        # voltage to fluorescence
        self.v2f = VoltageToFluorescenceConverter(v2f_specs)
        
        # cache for univariate spline
        self._spline_cache = dict()

    @cachedmethod(cache='_spline_cache')
    def get_spline(self, sweep_index) -> InterpolatedUnivariateSpline:
        return InterpolatedUnivariateSpline(
            x=self.ephys.time_list[sweep_index],
            y=self.ephys.voltage_list[sweep_index],
            k=1,  # linear
            ext=3,  # use bounadry values for extrapolation
        )
    
    def get_fluorescence(
            self,
            sweep_index: int,
            t: float,
            sampling_rate: float) -> np.ndarray:
        
        # get ephys spline
        spline = self.get_spline(sweep_index)
        
        # get resting membrane voltage
        v0 = self.ephys.voltage_list[sweep_index][0]
        
        # integrate fluorescence
        integ_times = np.linspace(t - 1. / sampling_rate, t, num=self.neuron_specs.n_integ_points)
        output_yx = np.zeros((self.height, self.width))
        for integ_time in integ_times:
            query_time_yx = integ_time - self.delay_mask
            query_voltage_raw_yx = spline(query_time_yx.flatten()).reshape(self.height, self.width)
            query_voltage_attenuated_yx = self.attenuation_mask * query_voltage_raw_yx + v0 * (1. - self.attenuation_mask)
            query_fluorescence_yx = self.v2f(query_voltage_attenuated_yx) * self.density
            output_yx += query_fluorescence_yx
        output_yx = output_yx / len(integ_times)
        return output_yx
