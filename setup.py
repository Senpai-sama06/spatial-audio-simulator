from setuptools import setup, find_packages

setup(
    name="spatial_audio_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pyroomacoustics",
        "soundfile",
        "h5py",
        "pydantic",
        "psutil",
        "kagglehub",
        "matplotlib",
        "librosa"
    ],
    author="DSP/Audio Core Engineering Team",
    description="High-fidelity spatial audio render for training Source Separation ML models.",
    python_requires=">=3.8",
)
