from collections.abc import Callable
from pathlib import Path
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class AudioFileDataset(Dataset):
    def __init__(
        self,
        dataset_index: list[tuple[Path, int]],
        loading_pipeline: Callable[[Path], torch.Tensor],
    ) -> None:
        """
        dataset_index: (audio_file_path, label_index)
        loading_pipeline: function that takes (file_path, target_sample_rate) and returns audio_tensor
        """

        self.dataset_items = dataset_index
        self.loading_pipeline = loading_pipeline

    def __len__(self) -> int:
        return len(self.dataset_items)

    def check_if_files_exist(self, n_jobs=-1) -> None:
        existing_dataset_items = []

        def _check_item(item):
            file_path, i = item
            if not file_path.exists():
                return None

            if file_path.stat().st_size <= 2 * 1024 * 1024:  # skip files <= 2 MiB
                return None
            # except Exception:
            #     return None
            # try:
            #     _ = self.loading_pipeline(file_path)
            # except Exception:
            #     return None
            return (file_path, i)

        existing_dataset_items = []
        
        for i in tqdm(range(len(self.dataset_items))):
            validated_item = self.dataset_items[i]
            if _check_item(validated_item) is not None:
                 existing_dataset_items.append(validated_item)   
            
        # results = tqdm(
        #     Parallel(n_jobs=n_jobs, backend='threading')(
        #         delayed(_check_item)(i) for i in self.dataset_items
        #     )
        # )
        # for r in results:
        #     if r is not None:
        #         existing_dataset_items.append(r)

        self.dataset_items = existing_dataset_items

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, label_index = self.dataset_items[index]

        audio_tensor = self.loading_pipeline(file_path)
        if audio_tensor is None:
            raise ValueError(f'Failed to load audio file: {file_path}')

        return audio_tensor, label_index
