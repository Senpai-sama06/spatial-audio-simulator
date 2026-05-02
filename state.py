import random
import numpy as np

class MasterState:
    _instance = None
    _seed_set = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MasterState, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize_seed(cls, seed: int):
        """
        Initializes the deterministic seed for all PRNGs.
        Must be called before any other processing.
        """
        cls.seed = seed
        cls._seed_set = True
        
        # Apply to global PRNGs
        random.seed(seed)
        np.random.seed(seed)
        
        # Pyroomacoustics relies on numpy's random state for some things,
        # but setting np.random.seed is sufficient for global scope.
        
        print(f"[STATE] Master Deterministic Seed initialized: {seed}")

    @classmethod
    def check_seed(cls):
        """
        Verifies that the seed was set, raising a fatal error if not.
        """
        if not cls._seed_set:
            raise RuntimeError("FATAL: Master seed not initialized. Pipeline cannot proceed without determinism.")
