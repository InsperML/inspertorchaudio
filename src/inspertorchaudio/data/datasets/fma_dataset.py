from functools import partial
from pathlib import Path
import audiofile
import pandas as pd
from typing import Literal

import torch
import torch.nn.functional as Fnn
import torchaudio
import torchaudio.functional as F

from .audio_dataset import AudioFileDataset
from .utils import AudioLoader

from sklearn.preprocessing import LabelEncoder

BLACKLIST = set([
    '108925',
    '113016',
    '113017',
    '113018', 
    '113019',
    '113020',
    # '11769',
    # '11776',
    # '011791',
    # '11763',
    # '20432',
    # '10577',
    # '10376',
    # '16878',
    # '16879',
    # '1929',
    # '26174',
    # '16880', 
    # '11794',
    # '17782',
    # '11787',
    # '27673',
    # '26169',
    # '11803',
    # '11793',
    # '11764',
])


def load_and_preprocess_tracks_csv(tracks_csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(
        tracks_csv_path,
        header=[0, 1],
        index_col=0,
    )
    df = df[[('track', 'genre_top'), ('set', 'split'), ('set', 'subset')]]
    df.columns = [c[1] for c in df.columns]
    return df


def make_filename(track_id: int, extension : str = 'wav') -> tuple[Path, Path]:
    track_filename = f'{track_id:06d}.{extension}'
    track_dir = track_filename[:3]
    return Path(track_dir), Path(track_filename)


def fma_dataset(
    tracks_csv_full_path: str | Path,
    audio_dir_full_path: str | Path,
    sample_length_seconds: float = 5.0,
    target_sample_rate: int = 16000,
    subset: Literal['small', 'medium', 'large'] = 'small',
    check_dataset_files:  bool = False,
):
    df = load_and_preprocess_tracks_csv(tracks_csv_full_path)
    df = df[df['subset'] == subset]
    
    blacklist_filter = ~df.index.astype(str).isin(BLACKLIST)
    df = df[blacklist_filter]

    df.loc[:, 'fullpath'] = df.index.to_series().apply(
        lambda track_id: audio_dir_full_path
        / make_filename(track_id)[0]
        / make_filename(track_id)[1]
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(df['genre_top'].unique())
    df['genre'] = label_encoder.transform(df['genre_top'])

    filter_train = df['split'] == 'training'
    filter_val = df['split'] == 'validation'
    filter_test = df['split'] == 'test'

    df_train = df[filter_train]
    df_val = df[filter_val]
    df_test = df[filter_test]

    audio_loader = AudioLoader(
        target_length_seconds=sample_length_seconds,
        target_sample_rate=target_sample_rate,
        normalize=True,
        convert_to_mono=True,
        lowpass_filter_width=6,
    )

    train_dataset = AudioFileDataset(
        dataset_index=list(zip(df_train['fullpath'], df_train['genre'])),
        loading_pipeline=audio_loader,
    )

    val_dataset = AudioFileDataset(
        dataset_index=list(zip(df_val['fullpath'], df_val['genre'])),
        loading_pipeline=audio_loader,
    )

    test_dataset = AudioFileDataset(
        dataset_index=list(zip(df_test['fullpath'], df_test['genre'])),
        loading_pipeline=audio_loader,
    )
    
    if check_dataset_files:
        print('Checking training dataset files...')
        train_dataset.check_if_files_exist()
        print('Checking validation dataset files...')
        val_dataset.check_if_files_exist()
        print('Checking test dataset files...')
        test_dataset.check_if_files_exist()

    return train_dataset, val_dataset, test_dataset, label_encoder
