from ._realized_moments import realized_moment


class RollingRealizedKurt_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, x):
        return realized_moment(x, self.w, self.sp, 4)
