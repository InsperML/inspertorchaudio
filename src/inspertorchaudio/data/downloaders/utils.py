from pathlib import Path

import requests
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
