import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_deformation(volume, sigma=4, grid=32):
    shape = volume.shape[-3:]

    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * grid
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * grid
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * grid

    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )

    indices = (x+dx, y+dy, z+dz)
    warped = map_coordinates(volume, indices, order=1, mode='reflect')
    return warped
