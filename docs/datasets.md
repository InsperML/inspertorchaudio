# Datasets

## Music Genre Classification

### FMA Small

    from inspertorchaudio.data import fma_dataset
    train_dataset, val_dataset, test_dataset, label_encoder = fma_dataset.get_fma_small_dataset()