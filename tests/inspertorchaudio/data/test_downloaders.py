from pathlib import Path

import pytest

from inspertorchaudio.data.downloaders.utils import download_file, unzip_file, download_dataset
import dotenv
import os

# download_file(target_url: str, target_path: str | Path) -> None

def test_download_file():
    dotenv.load_dotenv()
    download_dir = Path(os.getenv("DATA_DIR")).expanduser()
    target_url = "https://raw.githubusercontent.com/InsperML/inspertorchaudio/refs/heads/main/README.md"
    target_path = download_dir / "test/README.md"

    # Mock the download process
    download_file(target_url, target_path)

    # Check if the file was created
    assert target_path.exists()

def test_download_fma_metadata():
    path = download_dataset("fma_metadata", force_download=False)
    assert (Path(os.getenv("DATA_DIR")).expanduser() / "fma_metadata" / "fma_metadata.zip").exists()
    assert path.exists()
    assert isinstance(path, Path)

