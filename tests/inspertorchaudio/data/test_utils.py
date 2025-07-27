from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from inspertorchaudio.data.utils import file_loader, resampler


@dataclass
class ItemSpec:
    file_path: Path
    samples: np.ndarray
    sample_rate: int


@pytest.fixture
def audio_file_example() -> ItemSpec:
    return ItemSpec(
        file_path=Path('valid_file.wav'),
        samples=np.array([0.1, 0.2, 0.3]),
        sample_rate=44100,
    )


@pytest.fixture(autouse=True)
def mock_soundfile_read(monkeypatch, audio_file_example):
    """Mock soundfile.read to return a fixed audio tensor and sample rate."""

    def mock_read(file_path) -> tuple[np.ndarray, int] | None:
        if file_path != audio_file_example.file_path:
            raise sf.LibsndfileError(123)
        return (audio_file_example.samples, audio_file_example.sample_rate)

    monkeypatch.setattr('soundfile.read', mock_read)


def test_file_loader_valid_file(audio_file_example):
    """Test file_loader with a valid file."""
    result = file_loader(audio_file_example.file_path)

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2

    audio_tensor, sample_rate = result

    assert isinstance(audio_tensor, torch.Tensor)
    assert isinstance(sample_rate, int)

    assert torch.allclose(
        audio_tensor,
        torch.tensor(
            audio_file_example.samples,
            dtype=torch.float32,
        ),
    )
    assert audio_tensor.dtype == torch.float32
    assert sample_rate == audio_file_example.sample_rate


def test_file_loader_invalid_file():
    result = file_loader(Path('invalid_file.wav'))
    assert result is None


def test_file_loader_empty_file_path():
    result = file_loader(Path(''))
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

    expected_length = int(len(audio_tensor) * target_sample_rate / original_sample_rate)
    assert len(resampled_audio) == expected_length

    # Check that the resampled audio is a tensor
    assert isinstance(resampled_audio, torch.Tensor)
