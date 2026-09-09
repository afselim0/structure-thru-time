"""
Figure data: Gaussian vs Cauchy kernels.

Samples the two kernels on the distance grid the figure draws, and records the
distance range each one maps the "moderate similarity" band to -- the numbers the
figure annotates with arrows.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import out_dir, report, write_table

FIGURE = "GaussianvsCauchyKernels"

# the "moderate similarity" band on the y axis, and the sampling grid
Y_LOW, Y_HIGH = 0.05, 0.1
X_MIN, X_MAX, N_POINTS = 0, 12, 2000


def gaussian_inv(y):
    return np.sqrt(-2 * np.log(y))


def cauchy_inv(y):
    return np.sqrt(1 / y - 1)


def main():
    destination = out_dir(FIGURE)

    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    curves = pd.DataFrame({
        "x": x,
        "gaussian": np.exp(-x ** 2 / 2),
        "cauchy": 1.0 / (1.0 + x ** 2),
    })

    # the x range each kernel maps the band onto; the kernels are decreasing, so the
    # band's upper y bound gives the lower x bound
    bands = pd.DataFrame([
        {"kernel": "gaussian", "x_low": gaussian_inv(Y_HIGH), "x_high": gaussian_inv(Y_LOW)},
        {"kernel": "cauchy", "x_low": cauchy_inv(Y_HIGH), "x_high": cauchy_inv(Y_LOW)},
    ])

    params = pd.DataFrame([
        {"parameter": "y_low", "value": Y_LOW},
        {"parameter": "y_high", "value": Y_HIGH},
        {"parameter": "x_min", "value": X_MIN},
        {"parameter": "x_max", "value": X_MAX},
        {"parameter": "n_points", "value": N_POINTS},
    ])

    written = [
        write_table(curves, os.path.join(destination, "kernels.parquet")),
        write_table(bands, os.path.join(destination, "similarity_band.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
