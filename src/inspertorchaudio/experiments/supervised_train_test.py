import inspertorchaudio.data as data
import inspertorchaudio.data.fma_dataset as fma_dataset
import inspertorchaudio.models as models
import inspertorchaudio.models.dieleman2014 as dieleman2014
import inspertorchaudio.learning.supervised as supervised
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import torch


def fma_small_demo(
    experiment_name='baseline_fma',
    description='Baseline FMA small dataset with Dieleman2014 model',
    model_name='Dieleman2014',
):

    train_dataset, eval_dataset, test_dataset, label_encoder = (
        fma_dataset.get_fma_small_dataset(
            force_download=False,
            sample_length_seconds=5.0,
        )
    )
    
    n_classes = len(label_encoder.classes_)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        prefetch_factor=2,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        prefetch_factor=2,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        prefetch_factor=2,
    )

    lr = 1e-3

    backbone = dieleman2014.Dieleman2014(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=512,
        f_min=0.0,
        f_max=None,
        n_mels=128,
        power=1.0,
        compression_factor=10000,
        n_features_out=100,  # Adjusted to match the model's output
    )

    model = dieleman2014.DielemanClassifier(
        backbone=backbone,
        time_summarizer=None,  # No time summarization in this case
        n_classes=n_classes,  # Number of classes for classification
    )

    model.cuda()

    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
    )

    supervised.train_with_mlflow(
        experiment_name=experiment_name,
        description=description,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=100,
        use_eval=True,
        use_mlflow=True,
        model_name=model_name,
        patience_for_stop=10,
        lr_scheduler=lr_scheduler,
    )

