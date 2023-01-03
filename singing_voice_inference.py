import os
import torch

import vovit.display as vd
import vovit

from scipy.io.wavfile import  write
import matplotlib.pyplot as plt

data_path = '/home/jfm/singing_voice_sep_demo/splits'
dst_path = 'demo_samples/singing_voice_sep_results'
device = 'cuda:0'


sampler = vd.singing_voice_demo.DemoDataLoader(25, 16384, data_path, vd.t_dict)
N = len(sampler)
model = vovit.SingingVoiceVoViT(debug={}).to(device)
model.eval()

for idx in range(N):
    with torch.no_grad():
        key, kwargs = next(sampler)
        path = os.path.join(dst_path, key)
        if not os.path.exists(path):
            os.makedirs(path)
        mixture = sampler.load_audio(key, **kwargs).to(device)
        landmarks = sampler.load_landmarks(key, **kwargs).to(device)
        outputs = model.forward_unlimited(mixture, landmarks)

        # Dumping the results
        wav = outputs['estimated_wav'].squeeze().cpu().numpy()
        write(os.path.join(dst_path, f'{os.path.join(key, "estimated.wav")}'), 16384, wav)
        estimated_sp = torch.view_as_complex(outputs['estimated_sp']).squeeze().cpu().numpy()
        vd.plot_spectrogram(estimated_sp.squeeze(), 16384, 256, remove_labels=True)
        plt.tight_layout(True)
        plt.savefig(os.path.join(dst_path, f'{os.path.join(key, "estimated_sp.png")}'))
        print(f'[{idx}/{N}], {key}')