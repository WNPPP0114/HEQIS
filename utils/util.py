import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np


def setup_device(device):
    if isinstance(device, list) and len(device) == 1:
        device = torch.device(f'cuda:{device[0]}')
    else:
        device = None

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Apple Silicon
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    return device

def get_autocast_context(amp_dtype: str):
    if amp_dtype == "float16":
        return autocast(dtype=torch.float16)
    elif amp_dtype == "bfloat16":
        return autocast(dtype=torch.bfloat16)
    elif amp_dtype == "mixed":
        return autocast()
    else:
        # Return a dummy context manager, not using AMP
        from contextlib import nullcontext
        return nullcontext()


def compute_logdiff(arr):
    """
    Calculate ln((x_t - x_{t-1}) / x_{t-1}) for arr (shape=(T, D) or (T,)),
    pad the first point with 0 to prevent NaN/Inf.
    Return the same shape as the input.
    """
    eps = 1e-8
    a = np.array(arr, dtype=float)
    # Calculate difference and prevent division by zero
    diff = (a[1:] - a[:-1]) / (a[:-1] + eps)
    logdiff = np.log(diff + eps)
    # Pad the first time point
    if a.ndim == 1:
        pad = np.array([0.0])
    else:
        pad = np.zeros((1, a.shape[1]), dtype=float)
    out = np.concatenate([pad, logdiff], axis=0)

    # For simplicity, count the total number of values equal to 0 after replacement, and NaNs before replacement:
    nan_count = np.isnan(np.concatenate([pad, logdiff], axis=0)).sum()
    zero_count = (out == 0.0).sum()

    print(f"[compute_logdiff] NaN count (before replacement): {nan_count}, 0 value count (after replacement): {zero_count}")

    # Replace any NaN or Inf with 0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)