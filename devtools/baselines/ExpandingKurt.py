import pandas as pd


class ExpandingKurt_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().kurt().to_numpy()
