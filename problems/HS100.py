from typing import Tuple, Union, List, Type
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import number
from scipy.stats.qmc import LatinHypercube as lhs
from scipy.stats.qmc import scale

class CallableClass:
    def __call__(self, *args, **kwargs):
        pass


class HS100(CallableClass):
    r""".. math::
        \begin{align}
            \min\quad & (x_1 - 10)^2 + 5(x_2 - 12)^2 + x_3^4 + 3(x_4 - 11)^2\\
                        & + 10x_5^6 + 7x_6^2 + x_7^4 - 4x_6x_7 - 10x_6 - 8x_7\\[1em
            \text{s.t.}\quad & 2x_1^2 + 3x_2^4 + x_3 + 4x_4^2 + 5x_5 \leq 127\\
            & 7x_1 + 3x_2 + 10x_3^2 + x_4 - x_5 \leq 282\\
            & 23x_1 + x_2^2 + 6x_6 - 8x_7 \leq 196\\
            & 4x_1^2 + x_2^2 - 3x_1x_2 + 2x_3^2 + 5x_6 - 11x_7 \geq 0
        \end{align}

    The following bounds are placed on the variables:

    .. math::
        -10 \leq x_i \leq 10.075 \qquad i = 1,...,7
        
    This has a known solution of
    .. math::
        f(2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227)

    An initial guess is also defined for this class as

    .. math::
        f(1, 2, 0, 4, 0, 1, 1) = 714
    """

    name = 'hs100'

    def problem(self) -> dict:
        """Initialize the test problem"""
        return {
            "variables": {
            f"x{i+1}": {
                "type": "float",
                "bounds": [-10, 10.075],
                "shift": 0,
                "scale": 0.1,
            }
            for i in range(7)
        },
        "responses": {
            "f": {"type": "float", "shift": 0.0, "scale": 1e-3},
            "c1": {
                "type": "float",
                "bounds": [0, np.inf],
                "shift": 0,
                "scale": 1e-3,
            },
            "c2": {
                "type": "float",
                "bounds": [0, np.inf],
                "shift": 0,
                "scale": 1e-2,
            },
            "c3": {
                "type": "float",
                "bounds": [0, np.inf],
                "shift": 0,
                "scale": 1e-2,
            },
            "c4": {
                "type": "float",
                "bounds": [0, np.inf],
                "shift": 0,
                "scale": 1e-2,
            },
        },
        "objectives": ["f"],
        "constraints": ["c1", "c2", "c3", "c4"],
        }
    def known_solution(self) -> pd.Series:
        return pd.Series(
            data={
                "x1": 2.330499,
                "x2": 1.951372,
                "x3": -0.4775414,
                "x4": 4.365726,
                "x5": -0.6244870,
                "x6": 1.038131,
                "x7": 1.594227,
            }
        )
    def initial_guess(self) -> pd.Series:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        """
        return [1, 2, 0, 4, 0, 1, 1]

    def __call__(self, sites: pd.DataFrame) -> None:
        """Call to the HS100 function
        :param df: The dataframe that contains the input values, and is updated wit
        :type df: DataFrame
        """
        sites["f"] = (
            (sites.x1 - 10.0) * (sites.x1 - 10.0)
            + 5.0 * (sites.x2 - 12.0) * (sites.x2 - 12.0)
            + sites.x3 * sites.x3 * sites.x3 * sites.x3
            + 3.0 * (sites.x4 - 11.0) * (sites.x4 - 11.0)
            + 10.0 * sites.x5 * sites.x5 * sites.x5 * sites.x5 * sites.x5 * sites.x5
            + 7.0 * sites.x6 * sites.x6
            + sites.x7 * sites.x7 * sites.x7 * sites.x7
            - 4.0 * sites.x6 * sites.x7
            - 10.0 * sites.x6
            - 8.0 * sites.x7
        )
        sites["c1"] = (
            127.0
            - 2.0 * sites.x1 * sites.x1
            - 3.0 * sites.x2 * sites.x2 * sites.x2 * sites.x2
            - sites.x3
            - 4.0 * sites.x4 * sites.x4
            - 5.0 * sites.x5
        )
        sites["c2"] = (
            282.0
            - 7.0 * sites.x1
            - 3.0 * sites.x2
            - 10.0 * sites.x3 * sites.x3
            - sites.x4
            + sites.x5
        )
        sites["c3"] = (
            196.0
            - 23.0 * sites.x1
            - sites.x2 * sites.x2
            - 6.0 * sites.x6 * sites.x6
            + 8.0 * sites.x7
        )
        sites["c4"] = (
            -4.0 * sites.x1 * sites.x1
            - sites.x2 * sites.x2
            + 3.0 * sites.x1 * sites.x2
            - 2.0 * sites.x3 * sites.x3
            - 5.0 * sites.x6
            + 11.0 * sites.x7
        )
