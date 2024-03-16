import torch
from consource.dataloading.datasets.source_sep_dataset import SourceSepDataset
import pytorch_lightning as pl

class SourceSepDataModule(pl.LightningDataModule):
    
    def __init__(self, task = None, task_dir = None, target_sr=22050, target_class=None, target_len_s=6, batch_size=32):
        super().__init__()
        self.task = task
        self.task_dir = task_dir
        self.fetch_annotations = eval(f'self.fetch_{task}_annotations')
        self.target_sr = target_sr
        self.target_class = target_class
        self.target_len_s = target_len_s
        self.batch_size = batch_size
        
        self.annotations = self.fetch_annotations()
        self.train_annotations = self.annotations['train']
        self.val_annotations = self.annotations['val']
        self.test_annotations = self.annotations['test']
        
    def fetch_musdb_annotations(self):
        raise NotImplementedError("Musdb annotations not implemented yet")
    
    def fetch_musdbhq_annotations(self):
        raise NotImplementedError("MusdbHQ annotations not implemented yet")
    
    def fetch_medleydb_annotations(self):
        raise NotImplementedError("MedleyDB annotations not implemented yet")
    
    def fetch_moises_annotations(self):
        raise NotImplementedError("Moises annotations not implemented yet")
    
    def setup(self):
        self.train_dataset = SourceSepDataset(self.train_annotations, self.target_sr, self.target_class, self.target_len_s)
        self.val_dataset = SourceSepDataset(self.val_annotations, self.target_sr, self.target_class, self.target_len_s)
        self.test_dataset = SourceSepDataset(self.test_annotations, self.target_sr, self.target_class, self.target_len_s)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1)
        