import os
import zipfile
import importlib.resources
from pathlib import Path

import dotenv
import requests
import toml
from tqdm import tqdm


def download_file(target_url: str, target_path: str | Path) -> None:
    # Ensure the target directory exists
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(
            target_url,
            stream=True,
            timeout=10,
        )
    except requests.Timeout:
        raise RuntimeError(f'Timeout while trying to download {target_url}')
    except requests.RequestException as e:
        raise RuntimeError(f'Failed to download {target_url}: {e}')

    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with (
        open(target_path, 'wb') as f,
        tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=str(target_path),
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def unzip_file(zip_path: str | Path, extract_to: str | Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f'Extracted {zip_path} to {extract_to}')


def download_dataset(dataset_tag: str, force_download: bool = False) -> Path | None:
    """
    Downloads a dataset file.
    """

    data = toml.loads(
        importlib.resources.files('inspertorchaudio.resources')
        .joinpath('datasets.toml')
        .read_text()
    )

    if dataset_tag not in data:
        raise ValueError(f'Dataset {dataset_tag} not found in configuration file.')

    url = data[dataset_tag]['url']
    filename = url.split('/')[-1]

    dotenv.load_dotenv()
    data_dir_str = os.getenv('DATA_DIR')
    if not data_dir_str:
        raise RuntimeError('DATA_DIR environment variable is not set')

    download_dir = Path(data_dir_str).expanduser()
    local_dir = data[dataset_tag]['local_dir']
    target_path = download_dir / local_dir / filename

    if target_path.exists() and not force_download:
        print(f'File {target_path} already exists. Skipping download.')
        return target_path

    try:
        download_file(url, target_path)
        return target_path
    except Exception as e:
        print(
            f'Failed to download at {url} or extract file {filename} for dataset {dataset_tag}: {e}'
        )
        return None
