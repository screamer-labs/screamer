import talib


class PlusDM_talib:
    def __init__(self, window_size=14):
        self.w = window_size

    def __call__(self, high, low):
        return talib.PLUS_DM(high, low, timeperiod=self.w)
