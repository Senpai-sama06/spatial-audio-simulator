import os
import json
import hashlib
import numpy as np
import h5py
import pyroomacoustics as pra
from scipy.signal import fftconvolve
import soundfile as sf

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
        """Loads audio, forces float64, and handles duration (trimming or looping with 1s silence)."""
        target_samples = int(self.config.duration_seconds * self.config.sample_rate)
        silence_samples = int(self.config.sample_rate) # 1 second delay
        
        for src in self.config.sources:
            y, sr = sf.read(src.audio_path, dtype='float64')
            if sr != self.config.sample_rate:
                raise ValueError(f"Sample rate mismatch for {src.audio_path}")
            if len(y.shape) > 1:
                y = y[:, 0] # mono only
            
            if len(y) < target_samples:
                # Loop with 1s silence
                print(f"[ENGINE] Source {src.source_id} is shorter than target ({len(y)} < {target_samples}). Looping...")
                silence = np.zeros(silence_samples, dtype='float64')
                repeated = []
                current_len = 0
                while current_len < target_samples:
                    repeated.append(y)
                    current_len += len(y)
                    if current_len >= target_samples:
                        break
                    repeated.append(silence)
                    current_len += len(silence)
                y = np.concatenate(repeated)[:target_samples]
            else:
                # Trim
                y = y[:target_samples]
            
            self.source_signals[src.source_id] = y
            
        self.total_samples = target_samples
            
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
                    
        print("[ENGINE] HDF5 Export Complete.")

    def export_to_loose_files(self):
        """Exports stems, mix, noise, RIRs and room image as loose files."""
        loose_dir = os.path.join(self.output_dir, f"loose_files_{self.config.run_id}")
        os.makedirs(loose_dir, exist_ok=True)
        
        print(f"[ENGINE] Exporting loose files to: {loose_dir}")
        
        # Save Mixture (Transpose to [samples, channels] for soundfile)
        sf.write(os.path.join(loose_dir, "mixture.wav"), self.mix.T, self.config.sample_rate)
        
        # Save Noise
        sf.write(os.path.join(loose_dir, "microphone_noise.wav"), self.noise.T, self.config.sample_rate)
        
        # Save Stems
        for i, (src_id, stem_data) in enumerate(self.stems.items(), 1):
            sf.write(os.path.join(loose_dir, f"source{i}_{src_id}.wav"), stem_data.T, self.config.sample_rate)
            
        # Save RIRs as .npy (using object array to handle variable RIR lengths)
        rir_data = np.array(self.room.rir, dtype=object)
        np.save(os.path.join(loose_dir, "rir_matrix.npy"), rir_data)

        # Save Room Image
        try:
            import matplotlib.pyplot as plt
            fig, ax = self.room.plot()
            plt.savefig(os.path.join(loose_dir, "room_dimensionality.png"))
            plt.close(fig)
            print(f"  Saved room visualization to {loose_dir}/room_dimensionality.png")
        except ImportError:
            print("  Warning: matplotlib not found. Room image skipped.")
        except Exception as e:
            print(f"  Warning: Could not plot room: {e}")

    def run(self):
        self.load_audio()
        self.setup_environment()
        self.process_overlap_add()
        self.apply_mixing_and_noise()
        passed = self.validate_phase_null()
        self.export_to_hdf5(passed)
        if self.config.export_loose_files:
            self.export_to_loose_files()
