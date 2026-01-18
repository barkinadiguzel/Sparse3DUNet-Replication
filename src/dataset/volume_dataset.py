import numpy as np
from src.config import PATCH_SIZE, IN_CHANNELS

class VolumeDataset:
    def get_random_volume(self):
        return np.random.rand(IN_CHANNELS, *PATCH_SIZE).astype(np.float32)
