from ._bipower import bipower_and_realized


class RollingBipowerVar_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, x):
        return bipower_and_realized(x, self.w, self.sp)[:, 0]
