import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.io.wavfile import read, write
import vovit
import vovit.display as vd

device = 'cuda:0'
path = 'demo_samples/interview'
compute_landmarks = False
# if true landmarks are computed on-the-fly in the gpu, else it uses precomputed landmarks

# Loading the data
if compute_landmarks:
    speaker2_face = torch.from_numpy(np.load(f'{path}/speaker2.npy')).to(device)
    speaker1_face = torch.from_numpy(np.load(f'{path}/speaker1.npy')).to(device)
else:
    speaker1_face = torch.from_numpy(np.load(f'{path}/speaker1_ld.npy')).to(device)
    speaker2_face = torch.from_numpy(np.load(f'{path}/speaker2_ld.npy')).to(device)

mixture = torch.from_numpy(read(f'{path}/audio.wav')[1]).to(device)

print('Creating model instance...')
model = vovit.SpeechVoViT(extract_landmarks=compute_landmarks).to(device)
model.eval()
print('Done')

with torch.no_grad():
    print('Forwarding speaker1...')
    pred_s1 = model.forward_unlimited(mixture, speaker1_face)
    print('Forwarding speaker2...')
    pred_s2 = model.forward_unlimited(mixture, speaker2_face)

    wav_s1 = pred_s1['ref_est_wav'].squeeze().cpu().numpy()
    wav_s2 = pred_s2['ref_est_wav'].squeeze().cpu().numpy()
    vd.plot_spectrogram(pred_s1['ref_est_sp'].squeeze(), 16384, 256, remove_labels=True)
    plt.tight_layout(True)
    plt.savefig(f'{path}/s1_sp.png')
    vd.plot_spectrogram(pred_s2['ref_est_sp'].squeeze(), 16384, 256, remove_labels=True)
    plt.tight_layout(True)
    plt.savefig(f'{path}/s2_sp.png')
    write(f'{path}/speaker1_estimated.wav', vovit.core.AUDIO_SAMPLERATE, wav_s1)
    write(f'{path}/speaker2_estimated.wav', vovit.core.AUDIO_SAMPLERATE, wav_s2)

    vovit.utils.ffmpeg_join(f'{path}/speaker1_landmarks.mp4', f'{path}/speaker1_estimated.wav',
                            f'{path}/speaker1_estimated.mp4')
    vovit.utils.ffmpeg_join(f'{path}/speaker2_landmarks.mp4', f'{path}/speaker2_estimated.wav',
                            f'{path}/speaker2_estimated.mp4')
