import os

import torch

import inspertorchaudio.data.datasets.fma_dataset as fma_dataset
import inspertorchaudio.models.dieleman2014 as dieleman2014
import inspertorchaudio.learning.supervised as supervised_learning

from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path

FMA_DIRECTORY = Path("/mnt/data2/fma")
METADATA_SUBDIRECTORY = FMA_DIRECTORY / "fma_metadata"
TRACKS_CSV_PATH = METADATA_SUBDIRECTORY / "tracks.csv"

train_dataset, val_dataset, test_dataset, label_encoder = fma_dataset.fma_dataset(
    tracks_csv_full_path=TRACKS_CSV_PATH,
    audio_dir_full_path=FMA_DIRECTORY / "fma_wav16k",
    subset='medium',
    target_sample_rate=16000,
    check_dataset_files=True,
)

batch_size = 256
kwargs = {
     'num_workers' : 1,
     'pin_memory' : True,
     'prefetch_factor' : 3,
}

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

backbone = dieleman2014.Dieleman2014(
    sample_rate = 16000,
    n_fft = 1024,
    win_length = 1024,
    hop_length = 128,
    f_min = 10.0,
    f_max = 6000.0,
    n_mels = 128,
    power = 1.0,
    compression_factor = 1.0,
    n_features_out = 512,
)
n_classes = len(label_encoder.classes_)

classifier = dieleman2014.DielemanClassifier(
    backbone=backbone,
    n_classes=n_classes,
)

classifier.cuda()

optimizer = Adam(classifier.parameters(), lr=1e-4)

supervised_learning.train(
    model=classifier,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    epochs=1000,
    patience_for_stop=50,
    use_cuda=True,
    use_mlflow=False,
    use_eval=True,
)

print("Saving model...")
torch.save(classifier.state_dict(), "dieleman2014_fma_medium_2.pth")