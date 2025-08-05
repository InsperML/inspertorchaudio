"""
Baseline CNN model for audio classification.
Reference: Dieleman, S., & Schrauwen, B. (2014). End-to-end learning for music audio.
2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

Notes:
sr = 16kHz

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Dieleman2014(nn.Module):
    """
    Baseline CNN model for audio classification.
    Reference: Dieleman, S., & Schrauwen, B. (2014). End-to-end learning for music audio.
    2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    
    Default parameters are those from the best performing model in the paper.
    """
    def __init__(self,
                 sample_rate : int = 16000,
                 n_fft : int = 1024,
                 win_length : int = 256,
                 hop_length : int = 256,
                 f_min : float = 0.0,
                 f_max : float = None,
                 n_mels : int = 128,
                 power : float = 1.0,
                 compression_factor : float | None = 10000,
                 n_features_out : int = 100,
    ):
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length if win_length is not None else n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power
        )
        self.compression_factor = compression_factor
        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=32,
            kernel_size=8,
        )
        self.maxpool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=8,
        )
        self.maxpool2 = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(32, 50)
        self.fc2 = nn.Linear(50, n_features_out)

    def forward(self, x):
        """
        Forward pass of the CNN model.
        
        Args:
            x (torch.Tensor): Input audio waveform. Expected: batch x time
            
        Returns:
            torch.Tensor: Output of the CNN model. Dimensions: batch x time x n_features_out
        """

        # Apply mel spectrogram transformation
        x = self.melspectrogram(x)
        
        if self.compression_factor is not None:
            x = torch.log(1 + self.compression_factor * x)
        
        # Apply convolutional layers with max pooling
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x