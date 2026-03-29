"""Device selection and management for CPU/MPS inference."""

import os
import torch


def get_device(prefer_mps: bool = False) -> torch.device:
    """Select the best available device.

    Args:
        prefer_mps: If True, try to use Apple Metal (MPS) backend.
                    Default is False (CPU) for maximum compatibility.

    Returns:
        torch.device for inference.
    """
    if prefer_mps and torch.backends.mps.is_available():
        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    return torch.device("cpu")


def patch_cuda_references(module):
    """Recursively move all CUDA tensors in a module to CPU.

    Useful for loading pretrained models that were saved on CUDA.
    """
    for param in module.parameters():
        if param.is_cuda:
            param.data = param.data.cpu()
    for buf in module.buffers():
        if buf.is_cuda:
            buf.data = buf.data.cpu()
    return module


def cpu_map_location(storage, loc):
    """Map location function for torch.load to force CPU loading."""
    return storage
