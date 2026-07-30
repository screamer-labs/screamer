import talib


class MinusDI_talib:
    def __init__(self, window_size=14):
        self.w = window_size

    def __call__(self, high, low, close):
        return talib.MINUS_DI(high, low, close, timeperiod=self.w)
