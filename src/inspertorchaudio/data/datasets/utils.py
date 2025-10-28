import logging
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as Fnn
import torchaudio
import torchaudio.functional as F


class AudioLoader:
    # Utility class to load and preprocess audio files

    def __init__(
        self,
        target_length_seconds: float = 3.0,
        target_sample_rate: int = 16000,
        normalize: bool = True,
        convert_to_mono: bool = True,
        lowpass_filter_width: int = 6,
    ):
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.convert_to_mono = convert_to_mono
        self.target_length_seconds = target_length_seconds
        self.lowpass_filter_width = lowpass_filter_width

    def __call__(self, file_path: Path) -> torch.Tensor | None:
        audio_tensor, sample_rate = load_sample_and_convert_to_mono(
            file_path,
            length_seconds=self.target_length_seconds,
            to_mono=self.convert_to_mono,
        )

        if audio_tensor is None:
            raise ValueError(f'Failed to load audio file: {file_path}')

        if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
            F.resample(
                audio_tensor,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
                lowpass_filter_width=self.lowpass_filter_width,
            )
            sample_rate = self.target_sample_rate

        if self.target_length_seconds is not None:
            audio_tensor = resize(
                audio_tensor,
                sample_rate=self.target_sample_rate,
                target_length_seconds=self.target_length_seconds,
            )

        if self.normalize:
            audio_tensor_mean = audio_tensor.mean()
            audio_tensor_std = audio_tensor.std() + 1e-9
            audio_tensor = (audio_tensor - audio_tensor_mean) / audio_tensor_std

        return audio_tensor


def resize(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    target_length_seconds: float,
) -> torch.Tensor:
    target_length = int(target_length_seconds * sample_rate)
    if target_length is not None:
        if audio_tensor.size(0) < target_length:
            # Pad with zeros if shorter than target length
            padding = target_length - audio_tensor.size(0)
            audio_tensor = Fnn.pad(
                audio_tensor,
                (0, padding),
                mode='constant',
                value=0,
            )
        else:
            # Trim to target length if longer
            audio_tensor = audio_tensor[:target_length]
    return audio_tensor


def load_sample_and_convert_to_mono(
    file_path: Path,
    length_seconds: float,
    to_mono: bool = True,
    avoid_ends: float = 2.0,
) -> tuple[torch.Tensor, int]:
    """Loads a sample of specified length from an audio file, optionally converting it to mono.

    Args:
        file_path: Path to the audio file.
        length_seconds: Length of the audio sample to load in seconds.
        to_mono: Whether to convert the audio to mono.
        avoid_ends: Amount of seconds to avoid at the start and end of the audio file.

    Returns:
        A tuple containing the audio tensor and the sample rate.
    """
    if length_seconds <= 0:
        raise ValueError('length_seconds must be positive.')

    # Sample a random part of the audio file
    info = torchaudio.info(file_path, backend='soundfile')

    n_samples = info.num_frames
    sample_rate = info.sample_rate

    n_samples_to_load = int(length_seconds * sample_rate)

    n_samples_to_avoid = int(avoid_ends * sample_rate)
    if n_samples <= 2 * n_samples_to_avoid:
        raise ValueError(
            f'Audio file {file_path} is too short '
            f'to avoid {avoid_ends} seconds at start and end.'
        )

    available_samples = n_samples - 2 * n_samples_to_avoid
    if available_samples <= n_samples_to_load:
        raise ValueError(
            f'Audio file {file_path} is too short to load {n_samples_to_load} samples '
            f'and avoid {avoid_ends} seconds at start and end.'
        )

    lowest_start = n_samples_to_avoid
    highest_start = n_samples - (n_samples_to_load + n_samples_to_avoid) - 1
    start_sample = int(torch.randint(lowest_start, highest_start, (1,)).item())

    waveform, sample_rate = torchaudio.load(
        file_path,
        backend='soundfile',
        frame_offset=start_sample,
        num_frames=n_samples_to_load,
    )

    if waveform.shape[-1] < n_samples_to_load:
        raise ValueError(
            f'Audio file {file_path} is too short: {waveform.shape[-1]} samples. '
            f'File has {n_samples} samples, requested length is {n_samples_to_load} '
            f'samples, starting at sample {start_sample}. Info: {info}'
        )

    if to_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    waveform = waveform.squeeze()

    return waveform, sample_rate


def file_loader(file_path: Path) -> tuple[torch.Tensor, int] | None:
    try:
        data, sample_rate = sf.read(file_path)
    except sf.LibsndfileError as e:
        logging.error('Error reading file %s: %s', file_path, e)
        return None
    return torch.from_numpy(data).float(), sample_rate
