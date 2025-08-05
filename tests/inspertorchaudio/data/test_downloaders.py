import os
from pathlib import Path
import pytest
import dotenv

from inspertorchaudio.data.downloaders.utils import (
    download_dataset,
    download_file,
)


@pytest.fixture
def download_dir() -> Path:
    dotenv.load_dotenv()
    data_dir_str = os.getenv('DATA_DIR')
    if not data_dir_str:
        raise RuntimeError('DATA_DIR environment variable is not set')
    return Path(data_dir_str).expanduser()


def test_download_file(download_dir: Path) -> None:
    target_path = download_dir / 'test' / 'README.md'
    target_url = 'https://raw.githubusercontent.com/InsperML/inspertorchaudio/refs/heads/main/README.md'
    download_file(target_url, target_path)
    assert target_path.exists()


def test_download_fma_metadata(download_dir: Path) -> None:
    path = download_dataset('fma_metadata', force_download=False)
    assert path is not None
    assert isinstance(path, Path)
    assert path.exists()

    target_path = download_dir / 'fma_metadata' / 'fma_metadata.zip'
    assert target_path.exists()


