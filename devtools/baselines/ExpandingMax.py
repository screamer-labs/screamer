import pandas as pd


class ExpandingMax_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().max().to_numpy()
