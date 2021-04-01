from typing import NamedTuple, List, Tuple


class OptosynthSpecs(NamedTuple):
    width: int  # pixel
    height: int  # pixel
    sampling_rate: float  # Hz
    duration_per_segment: float  # s
    scale_factor: float  # relative to morphology data units
    n_neurons: int  # number of neurons in the FOV
    stim_amp_range_list: List[Tuple[float, float]]  # (lower, upper) bounds for stimulation current for each segment
    min_neuron_fluorescence_scale_factor: float  # minimum fluorescence scale factor for pasted neurons
    max_neuron_fluorescence_scale_factor: float  # maximum fluorescence scale factor for pasted neurons


class SyntheticNeuronSpecs(NamedTuple):
    dendritic_backprop_velocity: float  # pixel/sec
    dendritic_backprop_decay_lengthscale: float  # pixel
    min_reporter_density: float  # pixel^-2
    max_reporter_density: float  # pixel^-2
    reporter_density_var_lengthscale: float  # pixel
    ephys_lowpass_freq: float  # ephys lowpass frequency 


class VoltageToFluorescenceSpecs(NamedTuple):
    beta: float  # 1/mV
    v1: float  # low voltage
    f1: float  # fluorescence at v1
    v2: float  # high voltage
    f2: float  # fluorescence at v2


class BackgroundFluorescenceSpecs(NamedTuple):
    dynamic_n_components: int  # number of dynamic background components
    dynamic_x_lengthscale: float  # x variation lengthscale (pixels)
    dynamic_y_lengthscale: float  # y variation lengthscale (pixels)
    dynamic_fluorophore_density_scale: float  # dynamic background fluorescence density (pixel^-2)
    dynamic_temporal_frequency: float  # temporal scale of variation (Hz)
    static_x_lengthscale: float  # x variation lengthscale (pixels)
    static_y_lengthscale: float  # y variation lengthscale (pixels)
    static_min_total_fluorophore_density: float  # minimum total fluorescence density (pixel^-2)
    static_max_total_fluorophore_density: float  # maximum total fluorescence density (pixel^-2)

        
class CameraSpecs(NamedTuple):
    dc_offset: int  # in the units of camera readout
    gaussian_noise_std: float  # in the units of camera readout
    psf_lengthscale: int  # in the units of camera pixels
    readout_per_photon: float  # gain in readout per captured photon
    photon_per_fluorophore: float  # quantum yield x exposure time x absorption efficiency
