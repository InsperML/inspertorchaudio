# pylint: disable=missing-docstring
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F


def file_loader(file_path: Path) -> tuple[torch.Tensor, int] | None:
    try:
        data, sample_rate = sf.read(file_path)
    except sf.LibsndfileError as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    return torch.from_numpy(data).float(), sample_rate


def resampler(
    audio_tensor: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if original_sample_rate == target_sample_rate:
        return audio_tensor

    num_samples_original = audio_tensor.size(0)
    num_samples_target = int( \
        num_samples_original * target_sample_rate / original_sample_rate \
    )

    # Resample the audio tensor using linear interpolation
    resampled_audio = F.interpolate(
        audio_tensor.view(1, 1, -1),  # Add batch and channel dimensions
        size=num_samples_target,
        mode='linear',
        align_corners=False).view(-1)  # Remove batch and channel dimensions

    return resampled_audio
