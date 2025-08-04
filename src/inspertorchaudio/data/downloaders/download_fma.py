
from pathlib import Path

import pandas as pd

from .utils import (download_dataset, get_download_datadir, get_resources,
                    unzip_file)

def download_fma_small(force_download: bool = False) -> Path | None:
    """
    Downloads the FMA small dataset zip file and extracts it to the specified directory.
    Returns a path to the extracted directory.
    """
    download_dir = get_download_datadir()
    data = get_resources()
    local_dir = data['fma_small']['local_dir']
    target_file = download_dir / local_dir / 'fma_small.zip'

    if not target_file.exists() or force_download:
        zip_file = download_dataset('fma_small', force_download=force_download)
        if zip_file is None:
            raise RuntimeError('Failed to download the FMA small zip file.')
        unzip_file(zip_file, zip_file.parent)

    if not target_file.exists():
        raise FileNotFoundError(
            f'Expected file {target_file} not found after extraction.'
        )

    return target_file.parent

def get_fma_metadata_index(force_download: bool = False) -> None:
    """
    Downloads the FMA metadata zip file and extracts it to the specified directory.
    Returns a path to tracks.csv, which is the index file for the FMA dataset.
    """
    download_dir = get_download_datadir()
    data = get_resources()
    local_dir = data['fma_metadata']['local_dir']
    target_file = download_dir / local_dir / 'fma_metadata' / 'tracks.csv'

    if not target_file.exists() or force_download:
        zip_file = download_dataset('fma_metadata', force_download=force_download)
        if zip_file is None:
            raise RuntimeError('Failed to download the FMA metadata zip file.')
        unzip_file(zip_file, zip_file.parent)

    if not target_file.exists():
        raise FileNotFoundError(
            f'Expected file {target_file} not found after extraction.'
        )

    return target_file


def preprocess_fma_index(target_file: Path | str) -> pd.DataFrame:
    df = pd.read_csv(
        target_file,
        header=[0, 1, 2],
    )

    track_id = df[('Unnamed: 0_level_0', 'Unnamed: 0_level_1', 'track_id')]
    split = df[('set', 'split', 'Unnamed: 31_level_2')]
    subset = df[('set', 'subset', 'Unnamed: 32_level_2')]
    genre = df[('track', 'genre_top', 'Unnamed: 40_level_2')]

    meta_df = pd.DataFrame(
        {
            'track_id': track_id,
            'split': split,
            'subset': subset,
            'genre': genre,
        }
    )

    return meta_df
