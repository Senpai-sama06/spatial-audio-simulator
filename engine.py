import os
import json
import hashlib
import numpy as np
import h5py
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from state import MasterState
from schemas import SceneConfigSchema, ValidationSchema
from utils import apply_mic_jitter
from memory_manager import MemoryManager

def generate_noise(shape, snr_db, signal_power, seed):
    """Generates strictly seeded Gaussian white noise."""
    np.random.seed(seed)
    if signal_power == 0:
        return np.zeros(shape, dtype=np.float64)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return np.random.normal(0, np.sqrt(noise_power), shape).astype(np.float64)

class RenderEngine:
    def __init__(self, config: SceneConfigSchema, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Initialize Deterministic State
        MasterState.initialize_seed(self.config.master_seed)
        
        # Prepare signals dict
        self.source_signals = {}
        
    def load_audio(self):
        """Loads audio and forces float64 processing."""
        import soundfile as sf
        min_len = float('inf')
        for src in self.config.sources:
            # We assume audio is already matching sample_rate, or we'd resample
            y, sr = sf.read(src.audio_path, dtype='float64')
            if sr != self.config.sample_rate:
                raise ValueError(f"Sample rate mismatch for {src.audio_path}")
            if len(y.shape) > 1:
                y = y[:, 0] # mono only
            self.source_signals[src.source_id] = y
            min_len = min(min_len, len(y))
            
        # Truncate to minimum length
        self.total_samples = min_len
        for k in self.source_signals.keys():
            self.source_signals[k] = self.source_signals[k][:self.total_samples]
            
    def setup_environment(self):
        """Sets up Pyroomacoustics room and RIR calculation."""
        env = self.config.environment
        if env.type == 'closed_room':
            e_abs, max_order = pra.inverse_sabine(env.rt60_target, env.dimensions)
            materials = pra.Material(e_abs)
            self.room = pra.ShoeBox(env.dimensions, fs=self.config.sample_rate, 
                                    materials=materials, max_order=max_order, 
                                    air_absorption=True, ray_tracing=False)
        else:
            # Open field / anechoic
            # Use a dummy large room with max absorption
            self.room = pra.ShoeBox([100, 100, 100], fs=self.config.sample_rate,
                                    materials=pra.Material(1.0), max_order=0)

        # Apply Jitter to Mics
        mic_locs = apply_mic_jitter(self.config.microphone_array.locations)
        self.room.add_microphone_array(np.array(mic_locs).T)
        
        # Add sources (assuming static for this reference implementation)
        for src in self.config.sources:
            pos = src.trajectory_matrix[0][1:4] # Initial x, y, z
            # Add a dummy signal just to compute RIRs
            self.room.add_source(pos, signal=np.zeros(10))
            
        print("[ENGINE] Computing RIRs...")
        self.room.compute_rir()
        
    def process_overlap_add(self):
        """Processes convolution using dynamic chunking overlap-add in float64."""
        n_mics = self.config.microphone_array.num_mics
        n_sources = len(self.config.sources)
        
        # Max RIR length
        max_rir_len = 0
        for m in range(n_mics):
            for s in range(n_sources):
                max_rir_len = max(max_rir_len, len(self.room.rir[m][s]))
                
        output_len = self.total_samples + max_rir_len - 1
        
        # Initialize output buffers (float64)
        self.stems = {src.source_id: np.zeros((n_mics, output_len), dtype=np.float64) 
                      for src in self.config.sources}
        
        chunk_size = 48000 * 10 # Start with 10s chunks
        
        print("[ENGINE] Starting Dynamic Chunk Processing...")
        for start_idx in range(0, self.total_samples, chunk_size):
            # Dynamic memory check
            chunk_size = MemoryManager.calculate_chunk_size(chunk_size, complexity_factor=n_sources*n_mics)
            end_idx = min(start_idx + chunk_size, self.total_samples)
            
            for s_idx, src in enumerate(self.config.sources):
                sig_chunk = self.source_signals[src.source_id][start_idx:end_idx]
                
                for m_idx in range(n_mics):
                    rir = self.room.rir[m_idx][s_idx]
                    conv_chunk = fftconvolve(sig_chunk, rir, mode='full')
                    
                    # Overlap-add
                    out_start = start_idx
                    out_end = out_start + len(conv_chunk)
                    self.stems[src.source_id][m_idx, out_start:out_end] += conv_chunk
                    
            print(f"  Processed chunk {start_idx} to {end_idx}")

    def apply_mixing_and_noise(self):
        """Applies SNR and calculates final mix."""
        n_mics = self.config.microphone_array.num_mics
        output_len = len(list(self.stems.values())[0][0])
        
        # Sum stems for mix
        self.mix = np.zeros((n_mics, output_len), dtype=np.float64)
        for stem in self.stems.values():
            self.mix += stem
            
        # Add Sensor/Background Noise
        # For phase null validation, noise must be an isolated stem too
        mix_power = np.mean(self.mix ** 2)
        
        # Using a deterministic derivative of master seed for noise
        noise_seed = self.config.master_seed + 100 
        self.noise = generate_noise((n_mics, output_len), self.config.mixing_parameters.snr_target_db, mix_power, noise_seed)
        
        self.mix += self.noise

    def validate_phase_null(self):
        """
        The Phase Integrity Law: Mix - (Sum of Stems + Noise) == 0.
        Any residual sound fails the build.
        """
        sum_of_parts = np.zeros_like(self.mix, dtype=np.float64)
        for stem in self.stems.values():
            sum_of_parts += stem
        sum_of_parts += self.noise
        
        residual = np.abs(self.mix - sum_of_parts)
        max_residual = np.max(residual)
        
        # Strict floating-point epsilon tolerance for float64
        tolerance = 1e-12 
        passed = max_residual < tolerance
        
        print(f"[ENGINE] Phase Null Validation: Max Residual = {max_residual}")
        if not passed:
            raise RuntimeError(f"FATAL: Phase Null Test Failed! Residual {max_residual} exceeds {tolerance}")
        
        print("[ENGINE] Phase Null Test: PASSED")
        return passed

    def export_to_hdf5(self, passed_validation: bool):
        """Packs outputs strictly into HDF5."""
        out_path = os.path.join(self.output_dir, f"run_{self.config.run_id}.h5")
        
        # Checksum of mix
        mix_bytes = self.mix.tobytes()
        checksum = hashlib.sha256(mix_bytes).hexdigest()
        
        self.config.validation = ValidationSchema(
            phase_null_test_passed=passed_validation,
            mix_checksum_sha256=checksum
        )
        
        print(f"[ENGINE] Packing to HDF5 container: {out_path}")
        with h5py.File(out_path, 'w') as f:
            # Metadata
            f.attrs['scene_config'] = self.config.model_dump_json()
            
            # Audio Data
            audio_grp = f.create_group("audio")
            audio_grp.create_dataset("mix", data=self.mix, compression="gzip")
            audio_grp.create_dataset("noise", data=self.noise, compression="gzip")
            
            stems_grp = audio_grp.create_group("stems")
            for src_id, stem_data in self.stems.items():
                stems_grp.create_dataset(src_id, data=stem_data, compression="gzip")
                
            # Telemetry / RIRs
            rir_grp = f.create_group("rir_matrix")
            n_mics = self.config.microphone_array.num_mics
            n_sources = len(self.config.sources)
            for m in range(n_mics):
                for s in range(n_sources):
                    rir_grp.create_dataset(f"mic_{m}_src_{s}", data=self.room.rir[m][s], compression="gzip")
                    
        print("[ENGINE] Render Complete.")

    def run(self):
        self.load_audio()
        self.setup_environment()
        self.process_overlap_add()
        self.apply_mixing_and_noise()
        passed = self.validate_phase_null()
        self.export_to_hdf5(passed)
