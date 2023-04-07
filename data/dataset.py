import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df[['user_id', 'movie_id']].values[idx, :]
        y = self.df['implicit_feedback'].values[idx]
        return torch.tensor(X), torch.tensor(y)