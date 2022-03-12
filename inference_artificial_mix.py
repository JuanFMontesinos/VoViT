import torch
import numpy as np
from scipy.io.wavfile import read, write
from torch_mir_eval import bss_eval_sources
import vovit

device = 'cuda:0'

# Loading the data
tgt_face = torch.from_numpy(np.load('demo_samples/cRcvuz5Em8U/00227.npy')[:50]).to(device).unsqueeze(0)
audio2 = torch.from_numpy(
    vovit.utils.np_int2float(read('demo_samples/cRcvuz5Em8U/00227.wav')[1][:16384 * 2 - 1])).to(device)
audio2 /= audio2.abs().max()
audio1 = torch.from_numpy(
    vovit.utils.np_int2float(read('demo_samples/3r23tdRALns/00029.wav')[1][:16384 * 2 - 1])).to(device)
audio1 /= audio1.abs().max()
mixture = (audio1 + audio2).unsqueeze(0) / 2
model = vovit.End2EndVoViT(model_name='VoViT_speech', debug={}).to(device)
model.eval()
with torch.no_grad():
    pred = model(mixture, tgt_face)
print(bss_eval_sources(audio2.unsqueeze(0), pred['ref_est_wav'].unsqueeze(0)))
