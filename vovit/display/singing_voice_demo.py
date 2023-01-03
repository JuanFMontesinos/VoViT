import os
import numpy as np
import torch

from scipy.io.wavfile import read


t_dict = {'pRh9rKd2j64_0_15_to_0_55': {'initial_time': 15, 'sample_name': 'pRh9rKd2j64_0_15_to_0_55', 'n': 3},  # n=3
          'sEnTMgzw8ow_1_29_to_1_47': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},
          'sEnTMgzw8ow_1_5_to_2_07': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},
          'sEnTMgzw8ow_2_11_to_2_33': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},
          'sEnTMgzw8ow_2_38_to_2_53': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},
          'sEnTMgzw8ow_0_34_to_0_39': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},
          'Dyo7jzaCUhk_0_02_to_5_2': {'initial_time': 42, 'sample_name': '2_2', 'n': 3},
          # 'cttFanV0o7c_0_07_to_2_44': {'initial_time': 46, 'sample_name': 'top_right','n':1},  # 44-onwards llcp fails 8s
          'cttFanV0o7c_0_07_to_2_44': {'initial_time': 32, 'sample_name': 'bottom_left', 'n': 1},

          # Separates good from acmt + better than audio
          'vyu3HU3XWi4_0_3_to_0_4': {'initial_time': 0, 'sample_name': 'vyu3HU3XWi4_0_3_to_0_4', 'n': 1},
          'vyu3HU3XWi4_2_04_to_2_14': {'initial_time': 0, 'sample_name': 'vyu3HU3XWi4_2_04_to_2_14', 'n': 2},  # n=2
          'vyu3HU3XWi4_1_46_to_1_51': {'initial_time': 0, 'sample_name': 'vyu3HU3XWi4_1_46_to_1_51', 'n': 1},  # n=1
          'it6Ud6PDPes_2_22_to_2_27': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},  # n=1
          'SNgnylGkerE_0_15_to_0_2': {'initial_time': 0, 'sample_name': 'male_voice', 'n': 1},
          # n=1  # Unison appearance matters
          'WikcPREx0DM_2_12_to_2_17': {'initial_time': 0, 'sample_name': 'beatbox', 'n': 1},  # n=1
          'the_circle_of_life': {'initial_time': 2 * 60 + 55, 'sample_name': 'rafiki', 'n': 3},  # n=3
          'q9vqt-lwy3I_0_29_to_0_36': {'initial_time': 2, 'sample_name': 'lead_vocals', 'n': 1},  # n=1
          'q9vqt-lwy3I_1_33_to_1_43': {'initial_time': 2, 'sample_name': 'lead_vocals', 'n': 2},  # n=2
          'Gayh_GrCKgU_5_11_to_5_35': {'initial_time': 2, 'sample_name': 'lead_vocals', 'n': 1},
          # n=1, LLCP doesn't separate at all vocals not used
          'BtuwsjeN7Pk_4_22_to_4_28': {'initial_time': 2, 'sample_name': 'lead_vocals', 'n': 1},  # n=1
          'kce_zDH-OVA_0_43_to_0_5': {'initial_time': 3, 'sample_name': 'lead_vocals', 'n': 1},  # n=1
          'kce_zDH-OVA_1_42_to_1_5': {'initial_time': 0, 'sample_name': 'kce_zDH-OVA_1_42_to_1_5', 'n': 2},  # n=2
          'kce_zDH-OVA_2_09_to_2_3': {'initial_time': 8, 'sample_name': 'kce_zDH-OVA_2_09_to_2_3', 'n': 3},  # n=3
          'hWCkCSO8h9I_0_4_to_0_45': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 1},  # n=1
          '6Ws1WKA4z2k_0_35_to_0_48': {'initial_time': 0, 'sample_name': 'lead_vocals', 'n': 3},  # n=3
          }

class DemoDataLoader:
    def __init__(self, framerate: int, audiorate: int, data_path: str, dictionary={}):
        self.fps = framerate
        self.arate = audiorate
        self.data_path = data_path
        assert os.path.exists(data_path), f'The directorty {data_path} does not exist'
        self.core = dictionary
        self.generator = self._generator()

    def av_faces(self, video_id):
        path = os.path.join(self.data_path, 'frames', video_id)
        return os.listdir(path)

    def load(self, video_id, sample_name, initial_time, elements, n):
        output = {}
        for el in elements:
            loader = getattr(self, f'load_{el}')
            key = el if el != 'audio' else 'mixture'
            output[key] = loader(video_id, sample_name, initial_time, n)
        return output

    def load_frames(self, video_id, sample_name, initial_time, n):
        video_path = os.path.join(self.data_path, 'frames', video_id, sample_name) + '.npy'
        video = np.load(video_path)
        video = video[initial_time * self.fps:initial_time * self.fps + self.fps * 4 * n]
        return video

    def load_audio(self, video_id, sample_name, initial_time, n, reshape=False):
        audio_path = os.path.join(self.data_path, 'audio', video_id) + '.wav'
        audio = read(audio_path)[1][initial_time * self.arate:initial_time * self.arate + (self.arate * 4 - 1) * n]
        audio = torch.from_numpy(audio)
        audio = audio / audio.abs().max()
        if reshape:
            return audio.view(n, -1)
        return audio

    def load_landmarks(self, video_id, sample_name, initial_time, n, reshape=False):
        landmarks_path = os.path.join(self.data_path, 'landmarks', video_id, sample_name) + '.npy'
        landmarks = np.load(landmarks_path)[initial_time * self.fps:initial_time * self.fps + self.fps * 4 * n]

        landmarks = torch.from_numpy(landmarks)

        if not reshape:
            return landmarks
        landmarks = landmarks.reshape(n, -1, *landmarks.shape[1:])
        landmarks = landmarks.permute(0, 3, 1, 2)
        return landmarks.unsqueeze(-1).float()

    def get_sample(self, *args):
        key, kwargs = next(self)
        inputs = self.load(video_id=key, **kwargs, elements=args)
        return inputs, (key, kwargs)

    def _generator(self):
        for key, items in self.core.items():
            yield key, items

    def __len__(self):
        return len(self.core)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration:
            self.generator = self._generator()
            raise StopIteration