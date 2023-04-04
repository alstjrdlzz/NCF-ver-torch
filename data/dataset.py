from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocessing import preprocessor


class TrainDataset(Dataset):
    def __init__(self,
                 path: Path):
        raw_df = pd.read_csv(path/'u.data', sep='\t', 
                             encoding='latin-1', header=None)
        raw_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.implicit_df = preprocessor(raw_df)

    def __len__(self):
        return len(self.implicit_df)

    def __getitem__(self, idx):
        X = self.implicit_df[['user_id', 'movie_id']].values[idx, :]
        y = self.implicit_df['implicit_feedback'].values[idx]
        return torch.tensor(X), torch.tensor(y)