import torch
from consource.dataloading.datasets.source_sep_dataset import SourceSepDataset
import pytorch_lightning as pl
import pandas as pd
import os

class SourceSepDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 task = None,
                 task_dir = None,
                 target_sr=22050,
                 target_class=None,
                 target_len_s=6,
                 val_split=0):
        super().__init__()
        self.task = task
        self.task_dir = task_dir
        self.fetch_annotations = eval(f'self.fetch_{task}_annotations')
        self.target_sr = target_sr
        self.target_class = target_class
        self.target_len_s = target_len_s
        self.val_split = val_split
        
        self.annotations = self.fetch_annotations()
        self.train_annotations = self.annotations[self.annotations['split'] == 'train']
        self.val_annotations = self.annotations[self.annotations['split'] == 'val']
        self.test_annotations = self.annotations[self.annotations['split'] == 'test']
        
        # to dict
        self.train_annotations = self.train_annotations.to_dict('records')
        self.val_annotations = self.val_annotations.to_dict('records')
        self.test_annotations = self.test_annotations.to_dict('records')
        
    def fetch_musdb_annotations(self):
    
        folder_path = '/import/c4dm-datasets/MUSDB18HQ'
        
        dataframe = pd.DataFrame(columns=['folder_path', 'split'])
        
        # get train dataframe
        train_folder_path = os.path.join(folder_path, 'train')
        train_folders = os.listdir(train_folder_path)
        train_folders = [os.path.join(train_folder_path, f) for f in train_folders]
        train_df = pd.DataFrame(train_folders, columns=['folder_path'])
        train_df['split'] = 'train'
        
        #get test dataframe
        test_folder_path = os.path.join(folder_path, 'test')
        test_folders = os.listdir(test_folder_path)
        test_folders = [os.path.join(test_folder_path, f) for f in test_folders]
        test_df = pd.DataFrame(test_folders, columns=['folder_path'])
        test_df['split'] = 'test'
        
        if self.val_split > 0:
            val_df = test_df.sample(frac=self.val_split)
            test_df = test_df.drop(val_df.index)
            val_df['split'] = 'val'
            dataframe = pd.concat([train_df, test_df, val_df])
            
        else:
            dataframe = pd.concat([train_df, test_df])
        
        # create a dictionary of annotations for each split of format {folder : path to folder, stems : {class : path to stem}}
        all_stems = []
        for idx, row in dataframe.iterrows():
            folder = row['folder_path']
            stems = {}
            for stem in os.listdir(folder):
                if stem.endswith('.wav') and 'mixture' not in stem and 'accompaniment' not in stem:
                    class_name = stem.split('.')[0]
                    stems[class_name] = os.path.join(folder, stem)
            all_stems.append(stems)
        
        dataframe['stems'] = all_stems
        return dataframe
    
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
        