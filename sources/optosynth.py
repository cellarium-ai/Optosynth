import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
import logging
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import skvideo
from skvideo import io as skio
from PIL import Image, ImageOps, ImageDraw

from allen_data import \
    ProcessedAllenNeuronMorphology, \
    ProcessedAllenNeuronElectrophysiology

from specs import *
from bg import BackgroundFluorescenceGenerator
from camera import CameraSimulator
from neuron import SyntheticNeuronFluorescenceGenerator
from utils import \
    load_morphology, \
    load_electrophysiology, \
    scale_array, \
    generate_correlated_noise_1d, \
    generate_correlated_noise_2d, \
    paste_image


class Optosynth:
    def __init__(
            self,
            opto_specs: OptosynthSpecs,
            neuron_specs: SyntheticNeuronSpecs,
            bg_specs: BackgroundFluorescenceSpecs,
            v2f_specs: VoltageToFluorescenceSpecs,
            cam_specs: Optional[CameraSpecs],
            ephys_summary_df: pd.DataFrame,
            optosynth_data_path: str,
            device: torch.device,
            dtype: torch.dtype):
        
        self.opto_specs = opto_specs
        self.neuron_specs = neuron_specs
        self.bg_specs = bg_specs
        self.v2f_specs = v2f_specs
        self.cam_specs = cam_specs
        self.ephys_summary_df = ephys_summary_df
        self.optosynth_data_path = optosynth_data_path
        
        # at least one segment
        self.n_segments = len(opto_specs.stim_amp_range_list)
        assert self.n_segments > 0
        
        # time
        self.n_frames_per_segment = int(opto_specs.duration_per_segment * opto_specs.sampling_rate)
        self.time_per_segment = np.arange(0, self.n_frames_per_segment, dtype=np.float) / opto_specs.sampling_rate
        assert self.n_frames_per_segment > 0
        
        # generate experiment manifest
        self.manifest_df_list = self.generate_experiment_manifest(opto_specs, ephys_summary_df)
        self.selected_cell_ids = list(self.manifest_df_list[0]['cell_id'].values)
        
        # generate random location for somas
        self.xs_n = np.random.randint(0, opto_specs.width, size=opto_specs.n_neurons)
        self.ys_n = np.random.randint(0, opto_specs.height, size=opto_specs.n_neurons)
        
        # generate random global fluorescence scale factors for each neuron
        self.scale_n = (
            opto_specs.min_neuron_fluorescence_scale_factor +
            (opto_specs.max_neuron_fluorescence_scale_factor -
             opto_specs.min_neuron_fluorescence_scale_factor) *
                np.random.rand(opto_specs.n_neurons))
                
        self.log_info('Loading morphology and electrophysiology data ...')
        self.neurons = []
        for i_neuron in tqdm(range(self.opto_specs.n_neurons), position=0, leave=True):
            cell_id = self.selected_cell_ids[i_neuron]
            neuron = SyntheticNeuronFluorescenceGenerator(
                morph=load_morphology(optosynth_data_path, cell_id),
                ephys=load_electrophysiology(optosynth_data_path, cell_id),
                opto_specs=opto_specs,
                neuron_specs=neuron_specs,
                v2f_specs=v2f_specs,
                device=device,
                dtype=dtype)
            self.neurons.append(neuron)
        
        # instantiate the background generator
        self.log_info('Instantiating the background pattern generator ...')
        self.bg_gen = BackgroundFluorescenceGenerator(
            bg_specs=bg_specs,
            opto_specs=opto_specs,
            device=device,
            dtype=dtype)

        self.log_info('Instantiating the camera ...')
        if cam_specs is not None:
            self.camera = CameraSimulator(cam_specs)
        else:
            self.camera = None
        
        self.log_info('Generating ground truth masks ...')
        self.masks_nyx = np.zeros(
            (opto_specs.n_neurons, opto_specs.height, opto_specs.width), dtype=np.bool)
        for i_neuron in range(opto_specs.n_neurons):
            neuron = self.neurons[i_neuron]
            mask_yx = np.zeros((opto_specs.height, opto_specs.width), dtype=np.bool)
            paste_image(
                source_yx=neuron.mask,
                target_yx=mask_yx,
                source_anchor_x=int(neuron.xs),
                source_anchor_y=int(neuron.ys),
                target_anchor_x=self.xs_n[i_neuron],
                target_anchor_y=self.ys_n[i_neuron])
            self.masks_nyx[i_neuron, ...] = mask_yx

        self.bg_tyx_list = None
        self.neuron_fluorescence_tyx_list = None
        self.neuron_mean_fluorescence_nt_list = None
        self.clean_movie_tyx_list = None
        self.noisy_movie_tyx_list = None

    def generate_fluorescence_data(self):
        
        self.bg_tyx_list = []
        self.neuron_fluorescence_tyx_list = []
        self.neuron_mean_fluorescence_nt_list = []

        for i_segment in range(self.n_segments):
            self.log_info(f'Generating synthetic data for segment {i_segment + 1} / {self.n_segments} ...')
            segment_data = self._generate_segment_data(i_segment)
            self.bg_tyx_list.append(segment_data['bg_tyx'])
            self.neuron_fluorescence_tyx_list.append(segment_data['neuron_fluorescence_tyx'])
            self.neuron_mean_fluorescence_nt_list.append(segment_data['neuron_mean_fluorescence_nt'])
    
    def generate_camera_data(self):
        
        self.clean_movie_tyx_list = []
        self.noisy_movie_tyx_list = []
            
        for i_segment in range(self.n_segments):
            self.log_info(f'Generating clean and noisy camera readout for segment {i_segment + 1} / {self.n_segments} ...')
            camera_data = self._generate_camera_data(i_segment)
            self.clean_movie_tyx_list.append(camera_data['clean_movie_tyx'])
            self.noisy_movie_tyx_list.append(camera_data['noisy_movie_tyx'])

    def reset_camera(self, cam_specs: CameraSpecs):
        self.camera = CameraSimulator(cam_specs)
        
    def log_info(self, msg: str):
        print(msg)
    
    def generate_experiment_manifest(
            self,
            opto_specs: OptosynthSpecs,
            ephys_summary_df: pd.DataFrame) -> List[pd.DataFrame]:

        # step 1. select cell ids
        possible_cell_ids = set(ephys_summary_df['cell_id'].values)
        for (stim_amp_lo, stim_amp_hi) in opto_specs.stim_amp_range_list:
            possible_cell_ids = possible_cell_ids.intersection(
                ephys_summary_df[
                    (ephys_summary_df['stim_amp'] >= stim_amp_lo) &
                    (ephys_summary_df['stim_amp'] < stim_amp_hi)]['cell_id'].values)
        assert len(possible_cell_ids) >= opto_specs.n_neurons, \
            (f'Could not find enough neurons in the database to generate the '
             f'synthetic experiments; required: {opto_specs.n_neurons}; '
             f'found: {len(possible_cell_ids)}')
        self.log_info(
            f'Randomly choosing {opto_specs.n_neurons} from {len(possible_cell_ids)} '
            f'usable neurons ...')
        selected_cell_ids = np.random.choice(
            list(possible_cell_ids), size=opto_specs.n_neurons, replace=False).tolist()

        # step 2. generate the manifest
        manifest_df_list = []
        for (stim_amp_lo, stim_amp_hi) in opto_specs.stim_amp_range_list:
            manifest_df_list.append(
                pd.concat(
                    [ephys_summary_df[
                        (ephys_summary_df['cell_id'] == cell_id) &
                        (ephys_summary_df['stim_amp'] >= stim_amp_lo) &
                        (ephys_summary_df['stim_amp'] < stim_amp_hi)].sample()
                     for cell_id in selected_cell_ids],
                ignore_index=True))
        return manifest_df_list

    @staticmethod
    def _generate_fluorescence_data(
            time: np.ndarray,
            opto_specs: OptosynthSpecs,
            sweep_indices: List[int],
            neurons: List[SyntheticNeuronFluorescenceGenerator],
            xs_n: np.ndarray,
            ys_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_frames = len(time)
        neuron_mean_fluorescence_nt = np.zeros((opto_specs.n_neurons, n_frames))
        summed_fluorescence_tyx = np.zeros((n_frames, opto_specs.height, opto_specs.width))
        for i_t in tqdm(range(n_frames), position=0, leave=True):
            for i_neuron in range(opto_specs.n_neurons):
                neuron = neurons[i_neuron]
                fluorescence_yx = neuron.get_fluorescence(
                    sweep_index=sweep_indices[i_neuron],
                    t=time[i_t])
                neuron_mean_fluorescence_nt[i_neuron, i_t] = np.mean(fluorescence_yx[neuron.mask])
                paste_image(
                    source_yx=fluorescence_yx,
                    target_yx=summed_fluorescence_tyx[i_t, ...],
                    source_anchor_x=int(neuron.xs),
                    source_anchor_y=int(neuron.ys),
                    target_anchor_x=xs_n[i_neuron],
                    target_anchor_y=ys_n[i_neuron])

        return (
            neuron_mean_fluorescence_nt,
            summed_fluorescence_tyx)

    def _generate_segment_data(self, i_segment: int) -> Dict:
        # generate background
        bg_tyx = self.bg_gen.generate(
            n_frames=self.n_frames_per_segment,
            sampling_rate=self.opto_specs.sampling_rate)
        
        # get sweep indices for the segment
        sweep_indices = list(self.manifest_df_list[i_segment]['sweep_index'].values)
        
        # container for fluorescence from neurons
        neuron_fluorescence_tyx = np.zeros(
            (self.n_frames_per_segment, self.opto_specs.height, self.opto_specs.width))
        
        # container for individual traces
        neuron_mean_fluorescence_nt = np.zeros(
            (self.opto_specs.n_neurons, self.n_frames_per_segment))
        
        # generate fluorescence data
        (neuron_mean_fluorescence_nt,
         neuron_fluorescence_tyx ) = Optosynth._generate_fluorescence_data(
            time=self.time_per_segment,
            opto_specs=self.opto_specs,
            sweep_indices=sweep_indices,
            neurons=self.neurons,
            xs_n=self.xs_n,
            ys_n=self.ys_n)
        
        return {
            'bg_tyx': bg_tyx,
            'neuron_fluorescence_tyx': neuron_fluorescence_tyx,
            'neuron_mean_fluorescence_nt': neuron_mean_fluorescence_nt}
    
    def _generate_camera_data(self, i_segment: int) -> Dict:
        bg_tyx = self.bg_tyx_list[i_segment]
        neuron_fluorescence_tyx = self.neuron_fluorescence_tyx_list[i_segment]
        clean_movie_tyx = np.zeros(
            (self.n_frames_per_segment, self.opto_specs.height, self.opto_specs.width))
        noisy_movie_tyx = np.zeros(
            (self.n_frames_per_segment, self.opto_specs.height, self.opto_specs.width))
        for i_t in tqdm(range(self.n_frames_per_segment), position=0, leave=True):
            camera_out = self.camera(bg_tyx[i_t, ...] + neuron_fluorescence_tyx[i_t, ...])
            clean_movie_tyx[i_t, ...] = camera_out['clean_readout_yx']
            noisy_movie_tyx[i_t, ...] = camera_out['noisy_readout_yx']
            
        return {
            'clean_movie_tyx': clean_movie_tyx,
            'noisy_movie_tyx': noisy_movie_tyx,
        }
    
    def get_combined_mask(self, alpha: int = 50, show_soma: bool = True) -> Image:
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        combined_masks = Image.new('RGBA', size=(self.opto_specs.width, self.opto_specs.height))

        for i_neuron in range(self.opto_specs.n_neurons):
            c_mask = Image.fromarray(self.masks_nyx[i_neuron, ...])
            c_mask = ImageOps.colorize(
                c_mask.convert('L'),
                black='black',
                white=colors[i_neuron % len(colors)])
            c_mask = c_mask.convert('RGBA')
            c_pixels = c_mask.load()
            for x in range(self.opto_specs.width):
                for y in range(self.opto_specs.height):
                    if c_pixels[x, y] == (0, 0, 0, 255):
                        c_pixels[x, y] = (0, 0, 0, 0)
                    else:
                        c_pixels[x, y] = c_pixels[x, y][:3] + (alpha,)
            combined_masks.alpha_composite(c_mask)

        if show_soma:
            draw = ImageDraw.Draw(combined_masks)
            soma_center_rad = 2
            for x_s, y_s in zip(self.xs_n, self.ys_n):
                draw.ellipse([
                    (x_s - soma_center_rad, y_s - soma_center_rad),
                    (x_s + soma_center_rad, y_s + soma_center_rad)],
                fill=(255, 0, 0))

        return combined_masks

    def save(self, output_path: str):
        self.log_info(f'Saving data to {output_path} ...')

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        assert self.bg_tyx_list is not None
        assert self.neuron_fluorescence_tyx_list is not None
        assert self.neuron_mean_fluorescence_nt_list is not None
        assert self.clean_movie_tyx_list is not None
        assert self.noisy_movie_tyx_list is not None

        # masks
        np.save(
            os.path.join(output_path, 'masks_nyx.npy'),
            self.masks_nyx)
        
        # mean fluorescence per neuron
        np.save(
            os.path.join(output_path, 'neuron_mean_fluorescence_nt.npy'),
            np.concatenate(self.neuron_mean_fluorescence_nt_list, axis=1))
        
        # background
        np.save(
            os.path.join(output_path, 'background_tyx.npy'),
            np.concatenate(self.bg_tyx_list, axis=0))
        
        # clean fluorescence
        np.save(
            os.path.join(output_path, 'clean_fluorescence_tyx.npy'),
            np.concatenate(self.bg_tyx_list, axis=0) +
            np.concatenate(self.neuron_fluorescence_tyx_list, axis=0))

        # clean movie
        np.save(
            os.path.join(output_path, 'clean_movie_tyx.npy'),
            np.concatenate(self.clean_movie_tyx_list, axis=0))
        
        # noisy readout
        np.save(
            os.path.join(output_path, 'noisy_movie_tyx.npy'),
            np.concatenate(self.noisy_movie_tyx_list, axis=0))

        # soma coordinates
        np.save(
            os.path.join(output_path, 'soma_coords_n2.npy'),
            np.concatenate((self.xs_n[:, None], self.ys_n[:, None]), -1))
        
        # a snapshot of all neuron instances
        self.get_combined_mask().save(
            os.path.join(output_path, 'all_neurons.png'))

    def generate_avi_from_segment(
            self,
            segment_index: int,
            movie_type: str,
            output_root: str,
            n_frame_mean_subtraction: int):

        self.log_info(f'Saving {movie_type} .avi movie from segment {segment_index} to {output_root} ...')
        
        assert movie_type in {'clean', 'noisy'}

        # select movie
        if movie_type == 'clean':
            movie_tyx = self.clean_movie_tyx_list[segment_index]
            std_mult = 10.
        elif movie_type == 'noisy':
            movie_tyx = self.noisy_movie_tyx_list[segment_index]
            std_mult = 3.

        # normalize and generate
        movie_mean_yx = np.mean(movie_tyx[:n_frame_mean_subtraction], axis=0)
        res_movie_tyx = movie_tyx - movie_mean_yx[None, ...]
        res_movie_scale = std_mult * np.std(res_movie_tyx)    
        norm = Normalize(vmin=0, vmax=res_movie_scale, clip=True)
        normed_res_movie_tyx = (255 * norm(res_movie_tyx)[..., None])

        # write .avi
        writer = skvideo.io.FFmpegWriter(
            os.path.join(output_root, f"{movie_type}_movie_segment_{segment_index}.avi"),
            outputdict={'-vcodec': 'rawvideo', '-pix_fmt': 'yuv420p', '-r': '60'})
        for i_frame in range(len(normed_res_movie_tyx)):
            writer.writeFrame(normed_res_movie_tyx[i_frame])
        writer.close()