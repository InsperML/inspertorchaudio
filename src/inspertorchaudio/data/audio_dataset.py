from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset


class AudioFileDataset(Dataset):
    def __init__(
        self,
        dataset_items: list[tuple[Path, int]],
        target_sample_rate: int,
        file_loader: Callable[[Path], tuple[torch.Tensor, int] | None],
        resampler: Callable[[torch.Tensor, int, int], torch.Tensor],
        normalize: bool = True,
    ) -> None:
        '''
        dataset_items: (audio_file_path, label_index)
        file_loader: audio_file_path -> (audio_tensor, sample_rate)
        resampler: (audio_tensor, sample_rate, target_sample_rate) -> audio_tensor
        normalize: whether to normalize audio tensor to 0 mean and unit variance
        '''
        self.dataset_items = dataset_items
        self.target_sample_rate = target_sample_rate
        self.file_loader = file_loader
        self.resampler = resampler
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.dataset_items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, label_index = self.dataset_items[index]

        result = self.file_loader(file_path)
        if result is None:
            raise ValueError(f'Failed to load audio file: {file_path}')
        
        audio_tensor, sample_rate = result
        if audio_tensor is None:
            raise ValueError(f'Failed to load audio file: {file_path}')

        # Resample the audio tensor if necessary
        if sample_rate != self.target_sample_rate:
            audio_tensor = self.resampler(
                audio_tensor,
                sample_rate,
                self.target_sample_rate,
            )

        return audio_tensor, label_index
