from functools import partial
from pathlib import Path
import audiofile

import torch
import torch.nn.functional as Fnn
import torchaudio
import torchaudio.functional as F

from .audio_dataset import AudioFileDataset
from .downloaders.download_fma import (
    download_fma_small,
    get_fma_metadata_index,
    preprocess_fma_index,
)
from .utils import load_sample_and_to_mono, resample_and_resize
from .downloaders.utils import get_download_datadir, get_resources
from sklearn.preprocessing import LabelEncoder


def get_fma_small_dataset(
    force_download: bool = False, sample_length_seconds: float = 5.0
):
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
    df_small.loc[:, 'fullpath'] = df_small['path'].apply(lambda p: full_dir / p)

    label_encoder = LabelEncoder()
    label_encoder.fit(metadata_df['genre'].unique())
    df_small['genre'] = label_encoder.transform(df_small['genre'])

    filter_train = df_small['split'] == 'training'
    filter_val = df_small['split'] == 'validation'
    filter_test = df_small['split'] == 'test'

    df_train = df_small[filter_train]
    df_val = df_small[filter_val]
    df_test = df_small[filter_test]

    resample_fn = partial(
        resample_and_resize,
        target_length_seconds=sample_length_seconds,
        lowpass_filter_width=6,
    )
    load_fn = partial(
        load_sample_and_to_mono,
        length_seconds=sample_length_seconds,
    )

    train_dataset = AudioFileDataset(
        dataset_items=list(zip(df_train['fullpath'], df_train['genre'])),
        target_sample_rate=16000,
        file_loader=load_fn,
        resampler=resample_fn,
        normalize=True,
    )

    val_dataset = AudioFileDataset(
        dataset_items=list(zip(df_val['fullpath'], df_val['genre'])),
        target_sample_rate=16000,
        file_loader=load_fn,
        resampler=resample_fn,
        normalize=True,
    )

    test_dataset = AudioFileDataset(
        dataset_items=list(zip(df_test['fullpath'], df_test['genre'])),
        target_sample_rate=16000,
        file_loader=load_fn,
        resampler=resample_fn,
        normalize=True,
    )

    return train_dataset, val_dataset, test_dataset, label_encoder
