import os
from pathlib import Path

import dotenv

from inspertorchaudio.data.downloaders.utils import download_file
import pytest  # noqa
# download_file(target_url: str, target_path: str | Path) -> None


def test_download_file():
    dotenv.load_dotenv()

    data_dir_str = os.getenv('DATA_DIR')
    if not data_dir_str:
        raise RuntimeError('DATA_DIR environment variable is not set')

    download_dir = Path(data_dir_str).expanduser()
    target_path = download_dir / 'test' / 'README.md'

    target_url = 'https://raw.githubusercontent.com/InsperML/inspertorchaudio/refs/heads/main/README.md'

    # Mock the download process
    download_file(target_url, target_path)

    # Check if the file was created
    assert target_path.exists()
