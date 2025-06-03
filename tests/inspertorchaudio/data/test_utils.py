# pylint: disable=missing-docstring, redefined-outer-name, unused-argument
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from inspertorchaudio.data.utils import file_loader, resampler


@pytest.fixture
def mock_soundfile_read(monkeypatch):
    """Mock soundfile.read to return a fixed audio tensor and sample rate."""

    def mock_read(file_path):
        if file_path == Path("valid_file.wav"):
            return (np.array([0.1, 0.2, 0.3]), 44100)
        raise sf.LibsndfileError(123)

    monkeypatch.setattr("soundfile.read", mock_read)


def test_file_loader_valid_file(mock_soundfile_read):
    """Test file_loader with a valid file."""
    result = file_loader(Path("valid_file.wav"))

    assert result is not None

    assert isinstance(result, tuple)
    assert len(result) == 2

    audio_tensor, sample_rate = result

    assert isinstance(audio_tensor, torch.Tensor)
    assert isinstance(sample_rate, int)

    assert torch.allclose(audio_tensor, torch.tensor([0.1, 0.2, 0.3]))
    assert audio_tensor.dtype == torch.float32
    assert sample_rate == 44100


def test_file_loader_invalid_file(mock_soundfile_read):
    result = file_loader(Path("invalid_file.wav"))
    assert result is None


def test_file_loader_empty_file_path(mock_soundfile_read):
    result = file_loader(Path(""))
    assert result is None


def test_resampler_no_resampling():
    """Test resampler with no resampling needed."""
    audio_tensor = torch.tensor([0.1, 0.2, 0.3])
    original_sample_rate = 44100
    target_sample_rate = 44100

    resampled_audio = resampler(
        audio_tensor,
        original_sample_rate,
        target_sample_rate,
    )

    assert torch.equal(resampled_audio, audio_tensor)


def test_resampler_with_resampling():
    """Test resampler with resampling."""
    audio_tensor = torch.tensor([0.1, 0.2, 0.3])
    original_sample_rate = 44100
    target_sample_rate = 22050

    resampled_audio = resampler(
        audio_tensor,
        original_sample_rate,
        target_sample_rate,
    )

    expected_length = int(
        len(audio_tensor) * target_sample_rate / original_sample_rate)
    assert len(resampled_audio) == expected_length

    # Check that the resampled audio is a tensor
    assert isinstance(resampled_audio, torch.Tensor)
