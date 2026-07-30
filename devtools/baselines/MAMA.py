import talib


class MAMA_talib:
    def __init__(self, fast_limit=0.5, slow_limit=0.05):
        self.f, self.s = fast_limit, slow_limit

    def __call__(self, array):
        mama, _fama = talib.MAMA(array, fastlimit=self.f, slowlimit=self.s)
        return mama
