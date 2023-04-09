import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
        X = self.df[['user_id', 'movie_id']].values
        y = self.df['implicit_feedback'].values
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.X[idx, :]
        y = self.y[idx]
        return X, y