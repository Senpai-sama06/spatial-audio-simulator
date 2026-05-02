from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class EnvironmentType(str, Enum):
    closed_room = 'closed_room'
    open_field = 'open_field'

class SourceRole(str, Enum):
    target = 'target'
    interference = 'interference'
    background_noise = 'background_noise'

class SourceShape(str, Enum):
    point = 'point'
    spherical = 'spherical'

class SourceKinematics(str, Enum):
    static = 'static'
    moving = 'moving'

class EnvironmentSchema(BaseModel):
    type: EnvironmentType
    dimensions: Optional[List[float]] = Field(None, description="[x, y, z] Null if open_field")
    rt60_target: Optional[float] = Field(None, description="Null if open_field")
    speed_of_sound_m_s: float = 343.0

class MicrophoneArraySchema(BaseModel):
    num_mics: int
    locations: List[List[float]]

class SourceSchema(BaseModel):
    source_id: str
    role: SourceRole
    shape: SourceShape
    radius_m: Optional[float] = Field(None, description="Null if point source")
    kinematics: SourceKinematics
    trajectory_matrix: List[List[float]] = Field(
        ..., 
        description="[[t, x, y, z, v], ...]"
    )
    audio_path: str

class MixingParametersSchema(BaseModel):
    sir_target_db: float
    snr_target_db: float

class ValidationSchema(BaseModel):
    phase_null_test_passed: bool
    mix_checksum_sha256: str

class SceneConfigSchema(BaseModel):
    run_id: str
    master_seed: int
    sample_rate: int
    duration_seconds: float = Field(..., description="Target duration of the output audio in seconds")
    environment: EnvironmentSchema
    microphone_array: MicrophoneArraySchema
    sources: List[SourceSchema]
    mixing_parameters: MixingParametersSchema
    export_loose_files: bool = Field(False, description="If True, exports stems, mix, noise and RIRs as loose files in addition to HDF5")
    validation: Optional[ValidationSchema] = None
