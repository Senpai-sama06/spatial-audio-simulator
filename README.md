# Spatial Audio ML Data Rendering Engine

## 1. Concept & Philosophy
This engine is a high-fidelity, offline Digital Signal Processing (DSP) pipeline designed to generate training data for Source Separation Machine Learning models. 

Unlike real-time audio engines that prioritize speed, this system **sacrifices execution time for absolute mathematical precision**. It treats the rendering process as a scientific simulation rather than a playback event.

### Core Principles
*   **The Phase Integrity Law:** In the digital domain, a mixture must exactly equal the sum of its constituent stems and noise. Our pipeline enforces a "Phase Null" test where `Mix - (Sum of Stems + Noise) == 0`. Any non-zero residual triggers a fatal failure.
*   **Acoustic Accuracy:** We render sources *inside* the room. The "Ground Truth" for ML training includes the Room Impulse Response (RIR) and distance attenuation, ensuring the model learns to separate sources in realistic environments.
*   **Bit-Perfect Reproducibility:** Every run is governed by a master RNG seed. Given the same seed and config, the engine will produce the exact same bit-stream every time.
*   **Memory-First Architecture:** To handle massive datasets on varying hardware, the engine dynamically polls system RAM and keeps utilization below 80%, shrinking processing chunks as needed to avoid OS-level swapping.

---

## 2. Technical Features
*   **Windowed Sinc Interpolation:** Fractional delays for moving sources or geometry calculations use windowed sinc filters. Linear interpolation is strictly banned to prevent "zippering" artifacts and aliasing.
*   **Array Geometry Jitter:** To prevent ML models from "memorizing" specific microphone spacings, a Gaussian variance of $\pm 2\text{mm}$ is applied to all microphone coordinates for every scene.
*   **HDF5 Containerization:** Instead of thousands of loose `.wav` files, every render is packed into a high-performance HDF5 container (`.h5`) containing audio, RIR matrices, and telemetry.
*   **Double-Precision Processing:** All internal audio math is performed in `float64` to maintain precision across long convolution chains.

---

## 3. Installation & Setup

Ensure you have access to the specified Python virtual environment:

```bash
source /home/rpzrm/enigma/bin/activate
```

The pipeline requires the following dependencies (already installed in the environment):
*   `numpy` (Numerical processing)
*   `pyroomacoustics` (Acoustic simulation)
*   `pydantic` (Data schema validation)
*   `h5py` (Data containerization)
*   `scipy` (DSP utilities)
*   `psutil` (Hardware monitoring)

---

## 4. Usage

### Running a Test Simulation
To run a complete end-to-end rendering test with sample audio data:

```bash
python run_pipeline.py
```

### Configuration
The pipeline is driven by `SceneConfigSchema` (see `schemas.py`). A typical `scene_config.json` includes:
*   **Environment:** Room dimensions and $RT_{60}$ (reverberation time).
*   **Microphone Array:** Number of mics and their theoretical 3D coordinates.
*   **Sources:** Source roles (target/interference), shapes (point/spherical), and audio paths.
*   **Mixing Parameters:** Target Signal-to-Interference Ratio (SIR) and Signal-to-Noise Ratio (SNR).

### Outputs
Upon a successful run, the engine generates an `output/` directory containing:
1.  `run_[run_id].h5`: The master data container.
2.  `scene_config.json`: The metadata payload used for the run.
3.  `changelog.txt`: A history of architectural changes and reasoning.

---

## 5. Failure Protocols
If the pipeline encounters an issue, it will halt immediately:
*   **Phase Null Failure:** If the stems do not sum perfectly to the mix.
*   **Seed Error:** If the master seed is missing (ensuring no non-deterministic data is created).
*   **Memory Ceiling:** If the system cannot process even small chunks within the 80% RAM limit.
