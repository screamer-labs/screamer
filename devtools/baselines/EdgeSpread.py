from ._edge_spread import edge_rolling


class EdgeSpread_numpy:
    def __init__(self, window_size=20, sign=False, start_policy="strict"):
        self.w, self.sign, self.sp = window_size, sign, start_policy

    def __call__(self, open_, high, low, close):
        return edge_rolling(open_, high, low, close, self.w, self.sign, self.sp)
