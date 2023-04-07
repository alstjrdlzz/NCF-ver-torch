from torch.utils.data import DataLoader
from utils.util import load_data, split_data
from preprocessing import preprocessor
from data.dataset import TrainDataset


class TrainDataModule():
    def __init__(self, config):
        self.data_path = config['data_path']
        self.cv_strategy = config['cv_strategy']
        self.batch_size = config['batch_size']

        # load, preprocessing, split
        df = load_data(self.data_path)
        df = preprocessor(df)
        train_df, valid_df = split_data(df, self.cv_strategy)

        # dataset
        self.train_dataset = TrainDataset(train_df)
        self.valid_dataset = TrainDataset(valid_df)

        # dataloader
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, self.batch_size, shuffle=False)