import os

from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import imageio as io
import numpy as np
import librosa
import torch

import vovit

path = 'demo_samples/interview'
video_fps = 25
# =============================================================================
video = np.stack(io.mimread(path + '.mp4', memtest=False))
# Shape is (438, 720, 1280, 3)

plt.imshow(video[200])
plt.show()

# Crop the face for each speaker
speaker1 = video[:8 * video_fps, 100:450, 250:550]
speaker2 = video[:8 * video_fps, 50:400, 750:1100]

# Resize
s1 = torch.from_numpy(speaker1).permute(0, 3, 1, 2)
s2 = torch.from_numpy(speaker2).permute(0, 3, 1, 2)
T, H, W, C = speaker1.shape
speaker1 = torch.nn.functional.interpolate(s1.float(),
                                           scale_factor=224 / H, mode='bilinear').permute(0, 2, 3, 1).byte().numpy()
T, H, W, C = speaker2.shape
speaker2 = torch.nn.functional.interpolate(s2.float(),
                                           scale_factor=224 / H, mode='bilinear').permute(0, 2, 3, 1).byte().numpy()
# Saving the cropped faces as numpy arrays
if not os.path.exists(f'{path}'):
    os.mkdir(f'{path}')
np.save(f'{path}/speaker1.npy', speaker1)
np.save(f'{path}/speaker2.npy', speaker2)
io.mimwrite(f'{path}/speaker1.mp4', speaker1, fps=video_fps)
io.mimwrite(f'{path}/speaker2.mp4', speaker2, fps=video_fps)

audio, sr = librosa.load('demo_samples/interview.mp4', sr=vovit.core.AUDIO_SAMPLERATE, duration=9)
write('demo_samples/interview/audio.wav', sr, audio[:vovit.core.AUDIO_SAMPLERATE * 8])

vovit.utils.process_video(f'{path}/speaker1.mp4',
                          video_dst=f'{path}/speaker1_landmarks.mp4',
                          landmarks_dst=f'{path}/speaker1_ld.npy',
                          assert_fps=25
                          )
vovit.utils.process_video(f'{path}/speaker2.mp4',
                          video_dst=f'{path}/speaker2_landmarks.mp4',
                          landmarks_dst=f'{path}/speaker2_ld.npy',
                          assert_fps=25
                          )
