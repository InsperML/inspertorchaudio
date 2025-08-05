from inspertorchaudio.data.fma_dataset import get_fma_small_dataset
import pytest

def test_get_fma_small_dataset():
    train_dataset, val_dataset, test_dataset, encoder = get_fma_small_dataset(force_download=False)
    assert train_dataset is not None
    assert val_dataset is not None
    assert test_dataset is not None
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0
    # Check that the datasets are instances of AudioFileDataset
    from inspertorchaudio.data.audio_dataset import AudioFileDataset
    assert isinstance(train_dataset, AudioFileDataset)
    assert isinstance(val_dataset, AudioFileDataset)
    assert isinstance(test_dataset, AudioFileDataset)
    assert train_dataset[0][0].dim() == 1
    assert train_dataset[0][0].size(0) > 0
    assert isinstance(train_dataset[0][1], int)