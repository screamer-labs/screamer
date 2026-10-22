from ._spread_estimators import corwin_schultz


class CorwinSchultzSpread_numpy:
    def __init__(self, window_size=20, adjust_overnight=True, start_policy="strict"):
        self.w, self.adjust, self.sp = window_size, adjust_overnight, start_policy

    def __call__(self, high, low, close):
        return corwin_schultz(high, low, close, self.w, self.adjust, self.sp)
