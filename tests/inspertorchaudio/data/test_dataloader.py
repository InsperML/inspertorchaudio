from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from inspertorchaudio.data.dataloader import AudioFileDataset


@dataclass
class ItemSpec:
    audio_file_path: Path
    label_index: int
    duration: float
    sample_rate: int


@pytest.fixture
def audio_file_test_items():
    file_specs = [
        ItemSpec(
            audio_file_path=Path('test_data', 'short_audio.wav'),
            duration=1.0,
            sample_rate=44100,
            label_index=0,
        ),
        ItemSpec(
            audio_file_path=Path('test_data', 'long_audio.wav'),
            duration=10.0,
            sample_rate=16000,
            label_index=1,
        ),
        ItemSpec(
            audio_file_path=Path('test_data', 'medium_audio.wav'),
            duration=5.0,
            sample_rate=22050,
            label_index=2,
        ),
    ]
    return file_specs


def test_audio_file_dataset(audio_file_test_items):  # pylint: disable=redefined-outer-name
    dataset_items = [
        (item.audio_file_path, item.label_index) for item in audio_file_test_items
    ]

    mock_file_specs = {
        item.audio_file_path: (item.duration, item.sample_rate)
        for item in audio_file_test_items
    }

    def mock_file_loader(file_path: Path) -> tuple[torch.Tensor, int]:
        # Mock loading audio files by returning a tensor of zeros
        # and the sample rate specified in the test items.
        result = mock_file_specs.get(file_path)
        if result is None:
            return None

        duration, sample_rate = result
        audio_tensor = torch.zeros(int(sample_rate * duration))
        return audio_tensor, sample_rate

    def mock_resampler(
        audio_tensor: torch.Tensor,
        sample_rate: int,
        target_sample_rate: int,
    ) -> torch.Tensor:
        num_samples_original = audio_tensor.size(0)
        num_samples_target = int(num_samples_original * target_sample_rate / sample_rate)
        # Mock resampling by returning a tensor of zeros
        # with the target sample rate.
        return torch.zeros(num_samples_target)

    # Initialize the dataset
    dataset = AudioFileDataset(
        dataset_items,
        target_sample_rate=16000,
        file_loader=mock_file_loader,
        resampler=mock_resampler,
    )

    # Test iteration
    for audio_tensor, label_index in dataset:
        assert isinstance(audio_tensor, torch.Tensor)
        assert audio_tensor.dim() == 1
        assert audio_tensor.size(0) > 0
        assert isinstance(label_index, int)
