import pandas as pd


class ExpandingSkew_pandas:
    def __init__(self):
        pass

    def __call__(self, array):
        return pd.Series(array).expanding().skew().to_numpy()
