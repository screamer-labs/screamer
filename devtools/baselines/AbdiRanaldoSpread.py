from ._spread_estimators import abdi_ranaldo


class AbdiRanaldoSpread_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, high, low, close):
        return abdi_ranaldo(high, low, close, self.w, self.sp)
