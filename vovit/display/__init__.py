import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import torch


def plot_spectrogram(spectrogram, sr: int, hop_length,
                     title=None, remove_labels=False, remove_axis=False,
                     ax=None, log=False):
    if isinstance(spectrogram, torch.Tensor):
        if not spectrogram.is_complex():
            spectrogram = torch.view_as_complex(spectrogram.contiguous())
        spectrogram = spectrogram.detach().cpu().numpy()

    amplitude = librosa.amplitude_to_db(spectrogram, ref=np.max)
    if ax is None:
        fig, ax = plt.subplots()
    y_axis = 'log' if log else 'linear'
    librosa.display.specshow(amplitude, sr=sr, x_axis='time', y_axis=y_axis, ax=ax, cmap='magma',
                             hop_length=hop_length)
    if title is not None:
        ax.set_title(title)
    if remove_labels:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    if remove_axis:
        ax.set_axis_off()

    return fig, ax

