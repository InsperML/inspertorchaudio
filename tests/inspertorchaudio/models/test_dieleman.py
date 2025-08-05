from inspertorchaudio.models.dieleman2014 import DielemanClassifier, Dieleman2014
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_dieleman_classifier():
    """Test the DielemanClassifier with a simple forward pass."""
    # Create a dummy Dieleman2014 model
    backbone = Dieleman2014(
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

    # Create a DielemanClassifier with the dummy model
    classifier = DielemanClassifier(
        backbone=backbone,
        time_summarizer=None,  # No time summarization in this case
        n_classes=8,  # Number of classes for classification
    )

    # Create a dummy input tensor (batch size of 2, 16000 samples)
    x = torch.randn(2, 16000*5)

    # Forward pass through the classifier
    output = classifier(x)

    # Check the output shape
    assert output.shape == (2, 8)  # Batch size x number of classes