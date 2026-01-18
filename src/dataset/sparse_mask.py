import numpy as np

class SparseMask:
    def create_sparse_mask(self, shape, labeled_slices=[10, 30, 60]):
        mask = np.zeros(shape, dtype=np.int64)
        weights = np.zeros(shape, dtype=np.float32)

        for z in labeled_slices:
            mask[:, :, z] = np.random.randint(0, 3, size=shape[:2])
            weights[:, :, z] = 1.0

        return mask, weights
