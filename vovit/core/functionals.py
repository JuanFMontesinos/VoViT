import torch
__all__ = ['cast_dtype']



def cast_dtype(waveform: torch.tensor, raise_error: bool = False) -> torch.tensor:
    """
    Cast an audio array in integer format into float scaling properly .
    :param waveform: numpy array of an audio waveform in int format
    :type waveform: torch.tensor
    :param raise_error: Flag to raise an error if dtype is not int
    """
    if waveform.is_floating_point():
        return waveform
    if waveform.type() == 'torch.CharTensor':
        return (waveform / 128).float()

    elif waveform.type() == 'torch.ShortTensor':
        return (waveform / 32768).float()

    elif waveform.type() == 'torch.IntTensor':
        return (waveform / 2147483648).float()
    elif waveform.type() == 'torch.LongTensor':
        return (waveform / 9223372036854775808).float()
    elif raise_error:
        raise TypeError(f'  {waveform.type()} found')
    else:
        return waveform

