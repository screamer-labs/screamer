import pandas as pd


class ExpandingVar_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().var().to_numpy()
