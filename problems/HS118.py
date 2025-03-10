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
    
class HS118(CallableClass):
    """Implement the Hock-Schittkowski number 118 problem
    x0 = ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0,
    20.0 )
    f(x0) = 942.7162499999998
    x* = ( 8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0, 77
    f(x*) = 664.82045000
    """
    name = 'hs118'
    
    def problem(self) -> dict:
        """Initialize the test problem"""
        return {
            "constraints": [
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c6",
                "c7",
                "c8",
                "c9",
                "c10",
                "c11",
                "c12",
                "c13",
                "c14",
                "c15",
                "c16",
                "c17",
        ],
        "objectives": ["f"],
        "responses": {
            "f": {"scale": 0.001, "shift": 0.0, "type": "float"},
            "c1": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c2": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c3": {
                "bounds": [0.0, 14.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c4": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c5": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c6": {
                "bounds": [0.0, 14.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c7": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c8": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c9": {
                "bounds": [0.0, 14.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c10": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c11": {
                "bounds": [0.0, 13.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c12": {
                "bounds": [0.0, 14.0],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c13": {
                "bounds": [0.0, np.inf],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c14": {
                "bounds": [0.0, np.inf],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c15": {
                "bounds": [0.0, np.inf],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c16": {
                "bounds": [0.0, np.inf],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
            "c17": {
                "bounds": [0.0, np.inf],
                "scale": 0.1,
                "shift": 0.0,
                "type": "float",
            },
        },
        "variables": {
            "x1": {
                "bounds": [8.0, 21.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x2": {
                "bounds": [43.0, 57.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x3": {
                "bounds": [3.0, 16.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x4": {
                "bounds": [0.0, 90.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x5": {
                "bounds": [0.0, 120.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x6": {
                "bounds": [0.0, 60.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x7": {
                "bounds": [0.0, 90.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x8": {
                "bounds": [0.0, 120.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x9": {
                "bounds": [0.0, 60.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x10": {
                "bounds": [0.0, 90.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x11": {
                "bounds": [0.0, 120.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x12": {
                "bounds": [0.0, 60.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x13": {
                "bounds": [0.0, 90.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x14": {
                "bounds": [0.0, 120.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
            "x15": {
                "bounds": [0.0, 60.0],
                "scale": 0.01,
                "shift": 0.0,
                "type": "float",
            },
        },
    }
    def initial_guess(self) -> list:
        """Provide an initial guess for an optimizer
        :return: List providing the initial guess to use with an optimizer
        :rtype: list
        x0 = ( 20.0, 55.0, 15.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20.0, 20.0, 60.0, 20
        60.0, 20.0 )
        f(x0) = 942.7162499999998
        """
        return [
            20.0,
            55.0,
            15.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
            20.0,
            60.0,
            20.0,
        ]
    def known_solution(self) -> list:
        """
        Provide the known optimal solution.
        :return: List providing the variable values of the optimal solution of the
        :rtype: list
        x* = (8.0, 49.0, 3.0, 1.0, 56.0, 0.0, 1.0, 63.0, 6.0, 3.0, 70.0, 12.0, 5.0,
        f(x*) = 664.82045000
        """
        return [
            8.0,
            49.0,
            3.0,
            1.0,
            56.0,
            0.0,
            1.0,
            63.0,
            6.0,
            3.0,
            70.0,
            12.0,
            5.0,
            77.0,
            18.0,
        ]
    def __call__(self, sites: pd.DataFrame) -> None:
        """Call to the HS118 function
        :param df: The dataframe that contains the input values, and is updated wit
        :type df: DataFrame
        """
        sites["f"] = (
            2.3 * sites.x1
            + 1.0e-4 * sites.x1 * sites.x1
            + 1.7 * sites.x2
            + 1.0e-4 * sites.x2 * sites.x2
            + 2.2 * sites.x3
            + 1.5e-4 * sites.x3 * sites.x3
            + 2.3 * sites.x4
            + 1.0e-4 * sites.x4 * sites.x4
            + 1.7 * sites.x5
            + 1.0e-4 * sites.x5 * sites.x5
            + 2.2 * sites.x6
            + 1.5e-4 * sites.x6 * sites.x6
            + 2.3 * sites.x7
            + 1.0e-4 * sites.x7 * sites.x7
            + 1.7 * sites.x8
            + 1.0e-4 * sites.x8 * sites.x8
            + 2.2 * sites.x9
            + 1.5e-4 * sites.x9 * sites.x9
            + 2.3 * sites.x10
            + 1.0e-4 * sites.x10 * sites.x10
            + 1.7 * sites.x11
            + 1.0e-4 * sites.x11 * sites.x11
            + 2.2 * sites.x12
            + 1.5e-4 * sites.x12 * sites.x12
            + 2.3 * sites.x13
            + 1.0e-4 * sites.x13 * sites.x13
            + 1.7 * sites.x14
            + 1.0e-4 * sites.x14 * sites.x14
            + 2.2 * sites.x15
            + 1.5e-4 * sites.x15 * sites.x15
        )
        sites["c1"] = sites.x4 - sites.x1 + 7
        sites["c2"] = sites.x6 - sites.x3 + 7
        sites["c3"] = sites.x5 - sites.x2 + 7
        sites["c4"] = sites.x7 - sites.x4 + 7
        sites["c5"] = sites.x9 - sites.x6 + 7
        sites["c6"] = sites.x8 - sites.x5 + 7
        sites["c7"] = sites.x10 - sites.x7 + 7
        sites["c8"] = sites.x12 - sites.x9 + 7
        sites["c9"] = sites.x11 - sites.x8 + 7
        sites["c10"] = sites.x13 - sites.x10 + 7
        sites["c11"] = sites.x15 - sites.x12 + 7
        sites["c12"] = sites.x14 - sites.x11 + 7
        sites["c13"] = sites.x1 + sites.x2 + sites.x3 - 60.0
        sites["c14"] = sites.x4 + sites.x5 + sites.x6 - 50.0
        sites["c15"] = sites.x7 + sites.x8 + sites.x9 - 70.0
        sites["c16"] = sites.x10 + sites.x11 + sites.x12 - 85.0
        sites["c17"] = sites.x13 + sites.x14 + sites.x15 - 100.0
