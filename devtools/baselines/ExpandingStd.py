import pandas as pd


class ExpandingStd_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().std().to_numpy()
