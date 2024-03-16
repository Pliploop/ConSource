## create a pytorch dataset for source separation

import torch
import torchaudio
import numpy as np
import soundfile as sf
from consource.dataloading.loading_utils import *


class SourceSepDataset(torch.utils.data.Dataset):
    """
    Dataset class for source separation task.
    
    Args:
        annotations (dict): A dictionary containing the annotations for the dataset.
            It should have the following format:
            {
                'folder': 'path/to/folder',
                'stems': {
                    class : 'path/to/stem'
                }
            }
        target_sr (int, optional): The target sample rate for the audio. Defaults to 22050.
        target_class (str, optional): The target class to separate. If None, a random class will be selected. Defaults to None.
        target_len_s (int, optional): The target length of the audio in seconds. Defaults to 6.
    """
    
    def __init__(self, annotations, target_sr=22050, target_class=None, target_len_s=6):
        self.annotations = annotations
        self.target_sr = target_sr
        self.target_class = target_class
        self.target_n_samples = int(target_len_s * target_sr)
        self.target_len_s = target_len_s
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the audio samples and their corresponding labels.
                The dictionary has the following keys:
                - 'accomp': The mixed audio without the target class.
                - 'mixture': The mixed audio with the target class.
                - 'target': The audio of the target class.
                - 'target_class': The class label of the target audio.
        """
        folder = self.annotations[idx]['folder_path']
        stems = self.annotations[idx]['stems']
        
        # load the stems
        
        if self.target_class is not None:
            if self.target_class not in stems:
                print(f"Class {self.target_class} not found in {folder}")
                return self[idx+1]
            else:
                target_path = stems[self.target_class]
        else:
            # randomly select a target class
            target_class = np.random.choice(list(stems.keys()))
            target_path = stems[target_class]
            
        mix_paths = [stems[stem] for stem in stems if stem != target_class]
            
        # get a random start point based on the length of the target audio
        sr, frames = get_file_info(target_path)
        start = np.random.randint(0, frames - self.target_n_samples)
        
        target_audio = load_audio_chunk(target_path, self.target_sr, self.target_n_samples, start)
        accomp = torch.zeros_like(target_audio)
        for mix_path in mix_paths:
            accomp += load_audio_chunk(mix_path, self.target_sr, self.target_n_samples, start)
        mix = accomp + target_audio
        
        return {
            'accomp': accomp,
            'mix': mix,
            'target': target_audio,
            'target_class': target_class
        }