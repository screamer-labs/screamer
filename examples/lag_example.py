"""Lag a series: the same operator on a whole array and on a live stream."""
import numpy as np

import screamer
from screamer import Lag


def main():
    print("screamer", screamer.__version__)

    values = np.array([10.0, 20.0, 30.0, 40.0])

    # Batch: call the operator on the whole array.
    print("batch: ", Lag(2)(values))

    # Streaming: call a fresh operator on one value at a time.
    lag = Lag(2)
    print("stream:", [lag(v) for v in values])


if __name__ == "__main__":
    main()
