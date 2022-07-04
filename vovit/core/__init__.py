AUDIO_SAMPLERATE = 16384
VIDEO_FRAMERATE = 25
N_FFT = 1022
HOP_LENGTH = 256
SP_FREQ_SHAPE = N_FFT // 2 + 1

fourier_defaults = {"audio_samplerate": AUDIO_SAMPLERATE,
                    "n_fft": N_FFT,
                    "sp_freq_shape": SP_FREQ_SHAPE,
                    "hop_length": HOP_LENGTH}
core_path = __path__[0]

from .models import VoViT
