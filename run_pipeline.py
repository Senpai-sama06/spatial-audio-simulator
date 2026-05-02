import os
import json
import soundfile as sf
import numpy as np
from engine import RenderEngine
from schemas import SceneConfigSchema
import kagglehub

def download_sample_audio():
    # Helper to download short audio for testing pipeline
    path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
    files = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]
    return files[0], files[1]

def create_sample_config(audio_1, audio_2):
    return {
        "run_id": "test_run_001",
        "master_seed": 42,
        "sample_rate": 22050,  # LJSpeech is 22050
        "duration_seconds": 10.0,
        "environment": {
            "type": "closed_room",
            "dimensions": [5.0, 4.0, 3.0],
            "rt60_target": 0.3,
            "speed_of_sound_m_s": 343.0
        },
        "microphone_array": {
            "num_mics": 2,
            "locations": [[2.5, 2.0, 1.5], [2.6, 2.0, 1.5]]
        },
        "sources": [
            {
                "source_id": "target_1",
                "role": "target",
                "shape": "point",
                "radius_m": None,
                "kinematics": "static",
                "trajectory_matrix": [[0.0, 1.0, 1.0, 1.5, 0.0]],
                "audio_path": audio_1
            },
            {
                "source_id": "interference_1",
                "role": "interference",
                "shape": "point",
                "radius_m": None,
                "kinematics": "static",
                "trajectory_matrix": [[0.0, 4.0, 3.0, 1.5, 0.0]],
                "audio_path": audio_2
            }
        ],
        "mixing_parameters": {
            "sir_target_db": 5.0,
            "snr_target_db": 20.0
        },
        "export_loose_files": True
    }

if __name__ == "__main__":
    print("Downloading sample audio for pipeline test...")
    a1, a2 = download_sample_audio()
    
    config_dict = create_sample_config(a1, a2)
    config = SceneConfigSchema(**config_dict)
    
    # Save config to file as requested in docs (Metadata payload)
    os.makedirs("output", exist_ok=True)
    with open("output/scene_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    engine = RenderEngine(config, output_dir="output")
    engine.run()
    
    print("SUCCESS")
