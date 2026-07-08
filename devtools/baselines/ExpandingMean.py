import pandas as pd


class ExpandingMean_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().mean().to_numpy()
