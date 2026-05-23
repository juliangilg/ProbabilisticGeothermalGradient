import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pyproj import Transformer


class ColombianGeothermalGradient(Dataset):
    """
    Dataset for geothermal gradient estimation in Colombia.

    Parameters
    ----------
    data_dir : str
        Path to the CSV file.
    target_col : str
        Name of the target column.
    feature_cols : list or None
        List of feature columns. If None, all columns except target_col are used.
    transform : callable or None
        Optional transformation applied to the input features.
    target_transform : callable or None
        Optional transformation applied to the target.
    """

    def __init__(
        self,
        data_dir=None,
        target_col=None,
        feature_cols=None,
        coordinates_transform=None,
        scaley = False
    ):
        self.data = pd.read_csv(data_dir, index_col=0)
        self.target_col = target_col



        ## Convert latitude and longitude to EPSG:32618
        x_m, y_m = coordinates_transform.transform(self.data["Longitude"].values, self.data["Latitude"].values)
        self.data["x_m"] = x_m/1000
        self.data["y_m"] = y_m/1000

        if feature_cols is None:
            self.feature_cols = [col for col in self.data.columns if col != target_col]
        else:
            self.feature_cols = feature_cols


        self.X = self.data[self.feature_cols].values.astype(np.float32)
        if scaley:
            self.y = np.log(self.data[self.target_col].values.astype(np.float32))
        else:
          self.y = (self.data[self.target_col].values.astype(np.float32))

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


