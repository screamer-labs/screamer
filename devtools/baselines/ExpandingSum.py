import pandas as pd


class ExpandingSum_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().sum().to_numpy()
