import warnings
from collections.abc import Callable
from pathlib import Path
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore', module='libmpg123')


class AudioFileDataset(Dataset):
    def __init__(
        self,
        dataset_index: list[tuple[Path, int]],
        loading_pipeline: Callable[[Path], torch.Tensor],
    ) -> None:
        """
        Args:
            dataset_index: list of tuples (audio_file_path, label_index)
            loading_pipeline: function (file_path) -> audio_tensor
        """

        self.dataset_items = dataset_index
        self.loading_pipeline = loading_pipeline

    def __len__(self) -> int:
        return len(self.dataset_items)

    def check_if_files_exist(self) -> None:
        def check_item(file_path: Path, item_index: int) -> bool:
            if not file_path.exists():
                return False

            info = sf.info(file_path)
            if info.frames < info.samplerate:
                # skip files shorter than 1 second
                return False

            try:
                _ = self.__getitem__(item_index)
            except ValueError as v:
                print(f'Error loading file: {file_path}')
                print(f'Raised: {v}')
                return False

            return True

        def filter_valid_items() -> list[tuple[Path, int]]:
            valid_dataset_items: list[tuple[Path, int]] = []
            for dataset_item in tqdm(
                self.dataset_items,
                desc='Checking audio files existence',
            ):
                if check_item(*dataset_item):
                    valid_dataset_items.append(dataset_item)
            return valid_dataset_items

        self.dataset_items = filter_valid_items()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        file_path, label_index = self.dataset_items[index]

        try:
            audio_tensor = self.loading_pipeline(file_path)
        except Exception as e:
            raise ValueError(f'Error loading audio file {file_path}: {e}')

        if audio_tensor is None:
            raise ValueError(f'Failed to load audio file: {file_path} - loading pipeline returned None.')

        return audio_tensor, label_index
