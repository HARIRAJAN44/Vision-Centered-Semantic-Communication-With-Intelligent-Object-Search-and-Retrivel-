import torch
import numpy as np

def serialize_to_bits(q: torch.Tensor) -> np.ndarray:
    """
    Serializes a uint8 tensor to a 1D numpy array of bits (0s and 1s).
    """
    assert q.dtype == torch.uint8
    # Flatten to 1D array of bytes
    bytes_arr = q.cpu().numpy().flatten()
    # Unpack bytes to bits
    bits = np.unpackbits(bytes_arr)
    return bits

def deserialize_from_bits(bits: np.ndarray, shape: tuple) -> torch.Tensor:
    """
    Deserializes a 1D numpy array of bits back to a uint8 tensor of the given shape.
    """
    # Pack bits to bytes
    bytes_arr = np.packbits(bits)
    # Reshape and convert to tensor
    q = torch.from_numpy(bytes_arr).reshape(shape)
    return q
