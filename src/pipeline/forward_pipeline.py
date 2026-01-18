import torch

from src.model.unet3d import UNet3D
from src.dataset.volume_dataset import VolumeDataset
from src.dataset.sparse_mask import SparseMask
from src.loss.weighted_softmax import WeightedSoftmaxLoss


class ForwardPipeline:
    def __init__(self, model):
        self.model = model.eval()

    def run(self, volume):
        with torch.no_grad():
            x = torch.tensor(volume).unsqueeze(0).float()  
            out = self.model(x)
        return out


if __name__ == "__main__":
    # Dummy volume
    dataset = VolumeDataset()
    volume = dataset.get_random_volume()

    # Model
    model = UNet3D()
    pipeline = ForwardPipeline(model)

    # Forward pass
    output = pipeline.run(volume)

    # Sparse annotation
    mask_gen = SparseMask()
    targets, weights = mask_gen.create_sparse_mask(output.shape[-3:])

    targets = torch.tensor(targets).unsqueeze(0)
    weights = torch.tensor(weights).unsqueeze(0)

    # Loss
    loss_fn = WeightedSoftmaxLoss()
    loss = loss_fn(output, targets, weights)

    print("Output shape:", output.shape)
    print("Sparse loss:", loss.item())
