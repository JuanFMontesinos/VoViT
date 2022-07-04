import torch


def load_weights(path):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    keys = ["audio_processor._window", "audio_processor.sp2mel.fb", "audio_processor.mel2sp.fb", "ap._window",
            "ap.sp2mel.fb", "ap.mel2sp.fb", "audio_processor.wav2sp.window", "ap.wav2sp.window"]
    for key in keys:
        if key in state_dict:
            del state_dict[key]
    return state_dict

