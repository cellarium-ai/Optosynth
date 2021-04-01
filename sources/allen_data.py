import os
import pickle
import numpy as np
from typing import List, Tuple, Dict

from PIL import Image, ImageDraw

from allensdk.core.swc import Morphology
from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.ephys_extractor import EphysSweepFeatureExtractor


def get_bounding_box(
        morph: Morphology,
        incl_types: List[int] = [Morphology.DENDRITE, Morphology.SOMA]):
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    for m_type in incl_types:
        for n in morph.compartment_list_by_type(m_type):
            for c in morph.children_of(n):
                x0, y0 = n['x'], n['y']
                x1, y1 = c['x'], c['y']
                min_x = min(min_x, min(x0, x1))
                max_x = max(max_x, max(x0, x1))
                min_y = min(min_y, min(y0, y1))
                max_y = max(max_y, max(y0, y1))
    return min_x, min_y, max_x, max_y


class ProcessedAllenNeuronMorphology:
    def __init__(
            self,
            cell_id: int,
            global_scale_factor: float,
            dendrite_scale_factor: float,
            soma_scale_factor: float,
            mask: np.ndarray,
            soma_coords: Tuple[int, int],
            soma_radius: int):

        self.cell_id = cell_id
        self.global_scale_factor = global_scale_factor
        self.dendrite_scale_factor = dendrite_scale_factor
        self.soma_scale_factor = soma_scale_factor
        self.mask = mask
        self.soma_coords = soma_coords
        self.soma_radius = soma_radius
        
    @staticmethod
    def from_morphology(
            cell_id: int,
            morph: Morphology,
            global_scale_factor: float = 10.,
            dendrite_scale_factor: float = 2.0,
            soma_scale_factor: float = 1.0,
            soma_relative_wiggle_size: float = 0.2,
            soma_wiggle_n_components: int = 3) -> 'ProcessedAllenNeuronMorphology':
        
        mask_out = ProcessedAllenNeuronMorphology.generate_neuron_bitmask(
            morph,
            global_scale_factor,
            dendrite_scale_factor,
            soma_scale_factor,
            soma_relative_wiggle_size,
            soma_wiggle_n_components)
        
        return ProcessedAllenNeuronMorphology(
            cell_id=cell_id,
            global_scale_factor=global_scale_factor,
            dendrite_scale_factor=dendrite_scale_factor,
            soma_scale_factor=soma_scale_factor,
            mask=mask_out['mask'].astype(np.bool),
            soma_coords=mask_out['soma_coords'],
            soma_radius=mask_out['soma_radius'])

    @staticmethod
    def generate_neuron_bitmask(
            morph: Morphology,
            global_scale_factor: float,
            dendrite_scale_factor: float,
            soma_scale_factor: float,
            soma_relative_wiggle_size: float,
            soma_wiggle_n_components: int):
        min_x, min_y, max_x, max_y = get_bounding_box(morph)
        width = int(global_scale_factor * (max_x - min_x))
        height = int(global_scale_factor * (max_y - min_y))
        img = Image.new(mode='1', size=(width, height))
        draw = ImageDraw.Draw(img)

        def scale_x(val):
            return int(global_scale_factor * (val - min_x))

        def scale_y(val):
            return int(global_scale_factor * (val - min_y))

        def scale_rad(val1, val2):
            return int(max(1.0, 0.5 * global_scale_factor * dendrite_scale_factor * (val1 + val2)))
        
        def generate_polygon(x0, y0, radius, relative_wiggle_size, wiggle_n_components) -> List[Tuple]:
            x0, y0, radius = float(x0), float(y0), float(radius)
            phi_n_arr = 2 * np.pi * np.random.rand(wiggle_n_components)
            a_n_arr = np.random.randn(wiggle_n_components)
            a_n_arr[0] = 0.
            scale_factor = np.reciprocal(1e-3 + np.sum(np.abs(a_n_arr))) * relative_wiggle_size * radius
            a_n_arr = scale_factor * a_n_arr
            t_arr = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
            n_arr = np.arange(0, wiggle_n_components)
            radius_t = radius + np.sum(
                a_n_arr[None, :] * np.cos(n_arr[None, :] * t_arr[:, None] + phi_n_arr[None, :]),
                axis=-1)
            x_t = np.abs(radius_t) * np.cos(t_arr) + x0
            y_t = np.abs(radius_t) * np.sin(t_arr) + y0
            return [(int(x), int(y)) for x, y in zip(x_t, y_t)]

        for n in morph.compartment_list_by_type(Morphology.DENDRITE):
            for c in morph.children_of(n):
                x0, y0 = scale_x(n['x']), scale_y(n['y'])
                x1, y1 = scale_x(c['x']), scale_y(c['y'])
                rad = scale_rad(n['radius'], c['radius'])
                draw.line([(x0, y0), (x1, y1)], fill=1, width=rad, joint='curve')

        for n in morph.compartment_list_by_type(Morphology.SOMA):
            soma_x0, soma_y0 = scale_x(n['x']), scale_y(n['y'])
            soma_rad = int(np.ceil(global_scale_factor * soma_scale_factor * n['radius']))
            poly = generate_polygon(
                soma_x0, soma_y0, soma_rad,
                soma_relative_wiggle_size,
                soma_wiggle_n_components)
            draw.polygon(poly, fill=1)

        return {
            'mask': np.asarray(img.getdata()).reshape(height, width),
            'soma_coords': (soma_x0, soma_y0),
            'soma_radius': soma_rad
        }
    
    def save(self, output_path: str):
        with open(os.path.join(output_path, f'{self.cell_id}_processed_morphology.pkl'), 'wb') as f:
            pickle.dump(self.cell_id, f)
            pickle.dump(self.global_scale_factor, f)
            pickle.dump(self.dendrite_scale_factor, f)
            pickle.dump(self.soma_scale_factor, f)
            pickle.dump(self.mask, f)
            pickle.dump(self.soma_coords, f)
            pickle.dump(self.soma_radius, f)
            
    @staticmethod
    def from_file(input_file: str) -> 'ProcessedAllenNeuronMorphology':
        with open(input_file, 'rb') as f:
            loader = pickle.Unpickler(f)
            cell_id = loader.load()
            global_scale_factor = loader.load()
            dendrite_scale_factor = loader.load()
            soma_scale_factor = loader.load()
            mask = loader.load()
            soma_coords = loader.load()
            soma_radius = loader.load()
            
        return ProcessedAllenNeuronMorphology(
            cell_id=cell_id,
            global_scale_factor=global_scale_factor,
            dendrite_scale_factor=dendrite_scale_factor,
            soma_scale_factor=soma_scale_factor,
            mask=mask,
            soma_coords=soma_coords,
            soma_radius=soma_radius)


class ProcessedAllenNeuronElectrophysiology:
    def __init__(
            self,
            cell_id: int,
            current_list: List[np.ndarray],
            voltage_list: List[np.ndarray],
            time_list: List[np.ndarray],
            stim_amp_list: List[float],
            n_spikes_list: List[int],
            spike_features_list: List[List[Dict]]):

        self.cell_id = cell_id
        order = np.argsort(stim_amp_list)
        self.n_sweeps = len(current_list)
        self.stim_amp_list = [stim_amp_list[o] for o in order]
        self.current_list = [current_list[o] for o in order]
        self.voltage_list = [voltage_list[o] for o in order]
        self.time_list = [time_list[o] for o in order]
        self.stim_amp_list = [stim_amp_list[o] for o in order]
        self.n_spikes_list = [n_spikes_list[o] for o in order]
        self.spike_features_list = [spike_features_list[o] for o in order]

    @staticmethod
    def from_electrophysiology(
            cell_id: int,
            ephys: NwbDataSet,
            duration = 2.0) -> 'ProcessedAllenNeuronElectrophysiology':

        current_list = []
        voltage_list = []
        time_list = []
        stim_amp_list = []
        n_spikes_list = []
        spike_features_list = []

        for sweep_number in ephys.get_sweep_numbers():
            sweep_metadata = ephys.get_sweep_metadata(sweep_number)
            if sweep_metadata['aibs_stimulus_name'] == 'Long Square':
                sweep_data = ephys.get_sweep(sweep_number)
                amp = sweep_metadata['aibs_stimulus_amplitude_pa']
                index_range = sweep_data["index_range"]
                sampling_rate = sweep_data["sampling_rate"]
                current = sweep_data["stimulus"][index_range[0]:index_range[1] + 1]
                voltage = sweep_data["response"][index_range[0]:index_range[1] + 1]

                # truncate
                max_frames = int(duration * sampling_rate)
                assert max_frames < len(voltage)
                current = current[:max_frames] * 1e12  # in pA
                voltage = voltage[:max_frames] * 1e3  # in mV

                # extract featrures
                time = np.arange(0, max_frames, dtype=np.float) / sampling_rate  # in seconds
                ext = EphysSweepFeatureExtractor(
                    t=time,
                    v=voltage,
                    i=current)
                ext.process_spikes()
                spike_features = ext.spikes()
                n_spikes = len(spike_features)

                current_list.append(current)
                voltage_list.append(voltage)
                time_list.append(time)
                stim_amp_list.append(amp)
                n_spikes_list.append(n_spikes)
                spike_features_list.append(spike_features)
        
        return ProcessedAllenNeuronElectrophysiology(
            cell_id=cell_id,
            current_list=current_list,
            voltage_list=voltage_list,
            time_list=time_list,
            stim_amp_list=stim_amp_list,
            n_spikes_list=n_spikes_list,
            spike_features_list=spike_features_list)
        
    def save(self, output_path: str):
        with open(os.path.join(output_path, f'{self.cell_id}_processed_electrophysiology.pkl'), 'wb') as f:
            pickle.dump(self.cell_id, f)
            pickle.dump(self.current_list, f)
            pickle.dump(self.voltage_list, f)
            pickle.dump(self.time_list, f)
            pickle.dump(self.stim_amp_list, f)
            pickle.dump(self.n_spikes_list, f)
            pickle.dump(self.spike_features_list, f)
        
    @staticmethod
    def from_file(input_file: str) -> 'ProcessedAllenNeuronElectrophysiology':
        with open(input_file, 'rb') as f:
            loader = pickle.Unpickler(f)
            cell_id = loader.load()
            current_list = loader.load()
            voltage_list = loader.load()
            time_list = loader.load()
            stim_amp_list = loader.load()
            n_spikes_list = loader.load()
            spike_features_list = loader.load()
            
        return ProcessedAllenNeuronElectrophysiology(
            cell_id=cell_id,
            current_list=current_list,
            voltage_list=voltage_list,
            time_list=time_list,
            stim_amp_list=stim_amp_list,
            n_spikes_list=n_spikes_list,
            spike_features_list=spike_features_list)
