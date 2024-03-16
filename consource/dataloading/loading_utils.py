import soundfile as sf
import numpy as np
import torch
import torchaudio

def load_audio_chunk(path, target_sr, target_len_samples, start = None):
    sr, frames = get_file_info(path)
    new_target_len = int(target_len_samples * sr / target_sr)
    if start is not None:
        start = int(start * sr)
    else:
        start = np.random.randint(0, frames - new_target_len)
        
    # resample to target_sr
    audio, _ = sf.read(path, start = start, stop = start + new_target_len)
    audio = torchaudio.functional.resample(torch.tensor(audio).unsqueeze(0), sr, target_sr)
    return audio

def load_full_audio(path, target_sr):
    sr, _ = get_file_info(path)
    audio, _ = sf.read(path)
    audio = torchaudio.functional.resample(torch.tensor(audio), sr, target_sr)
    return audio
    
def get_file_info(path):
    info = sf.info(path)
    return info.samplerate, info.frames