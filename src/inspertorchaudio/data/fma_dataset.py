from functools import partial
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

from .audio_dataset import AudioFileDataset
from .downloaders.download_fma import (download_fma_small,
                                       get_fma_metadata_index,
                                       preprocess_fma_index)
from .downloaders.utils import get_download_datadir, get_resources
from sklearn.preprocessing import LabelEncoder

def load_and_to_mono(file_path: Path) -> tuple[torch.Tensor, int] | None:
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform.squeeze(0), sample_rate
    except Exception as e:
        print(f'Error loading {file_path}: {e}')
        return None

def get_fma_small_dataset(force_download: bool = False):
    
    fma_small_path = download_fma_small(force_download=force_download)
    if fma_small_path is None:
        raise RuntimeError('Failed to download or locate the FMA small dataset.')

    fma_metadata_path = get_fma_metadata_index(force_download=force_download)
    if fma_metadata_path is None:
        raise RuntimeError('Failed to download or locate the FMA metadata.')

    metadata_df = preprocess_fma_index(fma_metadata_path)
    

    
    filter_small = metadata_df['subset'] == 'small'
    download_dir = get_download_datadir()
    data = get_resources()
    local_dir = Path(data['fma_small']['local_dir'])
    full_dir = download_dir / local_dir / 'fma_small'


    df_small = metadata_df[filter_small].copy()
    df_small.loc[:,'fullpath'] = df_small['path'].apply(lambda p: full_dir / p)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(metadata_df['genre'].unique())
    df_small['genre'] = label_encoder.transform(df_small['genre'])
    
    filter_train = df_small['split'] == 'training'
    filter_val = df_small['split'] == 'validation'
    filter_test = df_small['split'] == 'test'
    
    df_train = df_small[filter_train]
    df_val = df_small[filter_val]
    df_test = df_small[filter_test ]

    resample_fn = partial(F.resample, lowpass_filter_width=6)

    train_dataset = AudioFileDataset(
        dataset_items=list(zip(df_train['fullpath'], df_train['genre'])),
        target_sample_rate=16000,
        file_loader=load_and_to_mono,
        resampler=resample_fn,
        normalize=True,
    )

    val_dataset = AudioFileDataset(
        dataset_items=list(zip(df_val['fullpath'], df_val['genre'])),
        target_sample_rate=16000,
        file_loader=load_and_to_mono,
        resampler=resample_fn,
        normalize=True,
    )

    test_dataset = AudioFileDataset(
        dataset_items=list(zip(df_test['fullpath'], df_test['genre'])),
        target_sample_rate=16000,
        file_loader=load_and_to_mono,
        resampler=resample_fn,
        normalize=True,
    )

    return train_dataset, val_dataset, test_dataset, label_encoder