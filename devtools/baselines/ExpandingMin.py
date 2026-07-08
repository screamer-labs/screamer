import pandas as pd


class ExpandingMin_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().min().to_numpy()
