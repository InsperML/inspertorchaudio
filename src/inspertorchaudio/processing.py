import logging
import math

import torch
import torchaudio


def _next_power_of_two(x):
    return 2 ** int(math.ceil(math.log2(x)))


def _get_mel_spectrogram_transform_params(
    audio_sample_rate_hz: int = 16000,
    window_length_secs: float | None = None,
    hop_length_secs: float | None = None,
    num_mel_bins: int = 128,
    lower_edge_hz: float = 0.0,
    upper_edge_hz: float | None = None,
) -> dict[str, float | str | None]:
    params = dict(
        sample_rate=audio_sample_rate_hz,
        n_mels=num_mel_bins,
        f_min=lower_edge_hz,
        f_max=upper_edge_hz,
        power=1.0,
        mel_scale='htk',
    )

    if window_length_secs is not None:
        window_length_samples = int(audio_sample_rate_hz * window_length_secs)
        fft_length_samples = _next_power_of_two(window_length_samples)
        params['win_length'] = window_length_samples
        params['n_fft'] = fft_length_samples

    if hop_length_secs is not None:
        hop_length_samples = int(audio_sample_rate_hz * hop_length_secs)
        params['hop_length'] = hop_length_samples

    return params


def get_mel_spectrogram_transform(
    audio_sample_rate_hz: int = 16000,
    window_length_secs: float | None = None,
    hop_length_secs: float | None = None,
    num_mel_bins: int = 128,
    lower_edge_hz: float = 0.0,
    upper_edge_hz: float | None = None,
):
    params = _get_mel_spectrogram_transform_params(
        audio_sample_rate_hz=audio_sample_rate_hz,
        window_length_secs=window_length_secs,
        hop_length_secs=hop_length_secs,
        num_mel_bins=num_mel_bins,
        lower_edge_hz=lower_edge_hz,
        upper_edge_hz=upper_edge_hz,
    )
    melspectrogram = torchaudio.transforms.MelSpectrogram(**params)  # type: ignore
    return melspectrogram


def frame(data, window_length, hop_length):
    num_samples = data.shape[-1]
    num_frames = 1 + (num_samples - window_length) // hop_length
    max_frames = num_frames * hop_length

    # Truncate to fit.
    data = data[..., :max_frames]

    # Add batch dimension if missing.
    if data.ndim == 2:
        data = data.unsqueeze(0)

    data_ = []

    logging.debug('Stacking')
    logging.debug(data.shape)

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        if end > num_samples:
            break
        data_.append(data[:, :, start:end])

    logging.debug(
        'Data to stack: %s, shape: %s, num_frames: %d, window_length: %d, hop_length: %d',
        data_[0].shape,
        len(data_),
        num_frames,
        window_length,
        hop_length,
    )

    data = torch.stack(data_, dim=1)  # Stack frames along the second dimension.

    logging.debug('Stacked data shape: %s', data.shape)

    return data
