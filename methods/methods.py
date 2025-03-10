from typing import Tuple, Union, List, Type
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import number
from scipy.stats.qmc import LatinHypercube as lhs
from scipy.stats.qmc import scale

class ConstraintCalculator:
    """
    A class that calculates the constraint violation of an optimization
    problem that is passed in.
    
    >>> myCalc = ConstraintCalculator(optprob)
    
    >>> myCalc = ConstraintCalculator(optprob, individual=True) # Return the
    overall constraint violation as well as each individual constraint
    violation.
    
    >>> viol = myCalc(data) # Here data is all points to calculate constraint
    violation of. We will only calculate the constraint violations of responses.
    """
    
    def __init__(
        self, problem: dict, individual: bool = False, method: str = "euclid"
    ) -> None:
        """Initialize the object.

        >>> myCalc = ConstraintCalculator(optprob)

        >>> myCalc = ConstraintCalculator(optprob, individual=True) # Return
        the overall constraint violation as well as each individual constraint
        violation.
        
        Parameters
        ----------
        problem : dict
            Dictionary containing information about the variables and responses
            of the optimization problem.
        individual : bool, optional
            Whether or not calls to the constraint calculator returns the
            violation for each constraint., by default False
        method : str, optional
            Aggregation method to use. Can be one of the following values, by
        default 'euclid'

        - ``euclid``: Euclidean/:math:`L_2` norm
            
            .. math::
                v = \\sqrt{x_1^2 + ... + x_n^2}
        
        - ``manhat``: Manhattan/taxicab/:math:`L_1` norm
            
            .. math::
                v = x_1 + ... + x_n

        - ``max``: Maximum/:math:`L_\\infty` norm
        
            .. math::
                v = \\max(x_1, ..., x_n)

        - ``avg``: Average violation value
            
            .. math::
                v = \\frac{x_1 + ... + x_n}{n}

        - ``median``: Median violation value

        - ``prod``: Product of violations
            
            .. math::
                v = x_1 * ... * x_n
        In each of these equations, :math:`x_i` represents the amount that
        a constraint deviates from its bounds in rescaled coordinates.
        """
        # Make sure a dictionary was passed in
        if not isinstance(problem, dict):
            raise TypeError(
                "ConstraintCalculator: opt_prob must be a "
                f"dictionary! Received a {type(problem).__name__} instead."
            )
        # Save optimization problem dictionary
        self._problem = problem
        # Create shift and scale object
        self._shift_scale = ShiftAndScale(problem)
        # Save constraint names
        self._constraint_names = problem["constraints"]
        # Save bounds in easy dataframe assembly format
        bounds = [
            [
                problem["responses"][constraint]["bounds"][0]
                for constraint in self._constraint_names
            ],
            [
                problem["responses"][constraint]["bounds"][1]
                for constraint in self._constraint_names
            ],
        ]
        # Shift and scale bounds
        rescaled_bounds = self._shift_scale.design_to_optimizer_space(
            pd.DataFrame(data=bounds, columns=self._constraint_names),
            self._constraint_names,
        )
        # Save bounds
        self._lower_bounds = {
            constraint: rescaled_bounds.loc[0, constraint]
            for constraint in self._constraint_names
        }
        
        self._upper_bounds = {
            constraint: rescaled_bounds.loc[1, constraint]
            for constraint in self._constraint_names
        }

        # Set whether or not each individual constraint violation is returned
        self._individual = individual
        
        # Set which method to use
        valid_methods = {
            "euclid", # Euclidean/L2 norm
            "manhat", # Manhattan/taxicab/L1 norm
            "max", # Maximium norm
            "avg", # Average violation
            "median", # Median violation
            "prod", # Product of violations
        }

        # Check that method is valid
        if method.lower() not in valid_methods:
            raise ValueError(
                f"ConstraintCalculator: {method} is not a valid "
                f'method! Must be one of: {", ".join(valid_methods)}'
            )
        
        self._method = method


    def __call__(
        self, data: pd.DataFrame
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """Calculate the constraint violation of the given points in rescaled coord
            
        Parameters
        ----------
        data : pd.DataFrame
            All data points to calculate the constraint violation of.
        
        Returns
        -------
        pd.Series or (pd.Series, pd.DataFrame)
            Constraint violation of each point. If `individual` was specified in
            the constructor then the constraint violation of each variable for
            all points is also returned.
        """
        # Create series to save violation of each point
        total_violation = pd.Series(index=data.index, dtype=np.float64)

        # Set violation of all points containing NaN values to NaN
        valid_vals = ~data.loc[:, self._constraint_names].isna().any(axis=1)
        total_violation.loc[~valid_vals] = np.nan

        # Shift and scale data
        rescaled_data = self._shift_scale.design_to_optimizer_space(
            data, self._constraint_names
        )

        # Calculate individual constraint violation
        violation = np.maximum(
            0,
            np.maximum(
            self._lower_bounds - rescaled_data, rescaled_data - self._upper_bounds
            ),
        )

        # Calculate total violation for each point
        total_violation.loc[valid_vals] = self._compute_violation(violation[valid_vals])
                                                                            
        # Return results
        if self._individual:
            return (total_violation, violation)
        return total_violation
    
    def _compute_violation(self, violations: pd.DataFrame) -> pd.Series:
        """Compute the total constraint violation using the desired method.

        Parameters
        ----------
        violations : pd.DataFrame
            Individual constraint violations.

        Returns
        -------
        pd.Series
            Constraint violation for each site.

        Raises
        -------
        NotImplementedError
            Method has not been implemented.
        """
        # Euclidean distance
        if self._method == "euclid":
            ret = ((violations**2).sum(axis=1)) ** 0.5
        # Manhattan/taxicab distance
        elif self._method == "manhat":
            ret = violations.sum(axis=1)
        # Maximum norm
        elif self._method == "max":
            ret = violations.max(axis=1)
        # Average violation
        elif self._method == "avg":
            ret = violations.mean(axis=1)
        # Median violation
        elif self._method == "median":
            ret = violations.median(axis=1)
        # Product of violations
        elif self._method == "prod":
            ret = violations.prod(axis=1)
        else:
            raise NotImplementedError(
                f"ConstraintCalculator: {self._method} " "has not been implemented"
            )
        return ret

class ShiftAndScale:
    """Used to get variable and response values into the same order of
    magnitude to yieldK better optimization results.
    """

    def __init__(self, bound_prob: dict) -> None:
        """Constructor method
        :param bound_prob: Optimization problem defined as dictionary. Used to
        gather information about variables and responses. The shift value
        will be added to the value and scale will be multiplied to it.
        Missing or None values are assumed to be zero or one for shift and
        scale, respectively.
        :type bound_prob: dict
        """
        # Get shift and scale values for variables and responses
        var_shift, var_scale = self._dict_to_series(bound_prob["variables"])
        res_shift, res_scale = self._dict_to_series(bound_prob["responses"])
        # Store values into series to make arithmetic easier
        self.shift = pd.concat((var_shift, res_shift))
        self.scale = pd.concat((var_scale, res_scale))
        self._all = list(self.shift.index)
        
    def design_to_optimizer_space(
        self, points: pd.DataFrame, cols: List[str] = None, suffix: str = None
    ) -> pd.DataFrame:
        """Perform the shift and scaling on specified columns of provided
        points. Rescaled columns can also have their name appended with an
        optional suffix.
        :param points: Data points containing the points to shift and scale.
        :type points: DataFrame
        :param cols: (Optional) List of columns to perform the rescale on.
        :type cols: list
        :param suffix: (Optional) Value to append at end of column names.
        :type suffix: str
        :return: A dataframe containing the columns the operation was performed
        on with the shifted and scaled values.
        :rtype: DataFrame
        """
        if cols is None:
            cols = self._all
        # Get requested columns and perform shift and scale
        ret = points.loc[:, points.columns.isin(cols)]
        ret += self.shift.loc[self.shift.index.isin(cols)]
        ret *= self.scale.loc[self.scale.index.isin(cols)]
        # Append suffix if given
        if suffix:
            ret.columns += suffix
        # Return result
        return ret
    def optimizer_to_design_space(
        self, points: pd.DataFrame, cols: List[str] = None
    ) -> pd.DataFrame:
        """Perform the transformation from Optimizer Space to Design Space
        :param points: Data points containing the points in Optimizer Space that wi
        :type points: DataFrame
        :param cols: (Optional) List of columns to perform the transformation on.
        :type cols: list
        :return: A dataframe containing the unshifted and unscaled columns, that is
        :rtype: DataFrame
        """
        if cols is None:
            cols = self._all
        # Get requested columns and perform shift and scale
        ret = points.loc[:, points.columns.isin(cols)]
        ret /= self.scale.loc[self.scale.index.isin(cols)]
        ret -= self.shift.loc[self.shift.index.isin(cols)]
        # Return result
        return ret
    def to_optimizer_problem(self, problem: dict) -> dict:
        """Return the optimization problem in Optimizer Space
        This method takes an optimization problem in the main space
        and shifts and scales it
        Parameters
        ----------
        problem : dict
        The problem in the Design Space
        Returns
        -------
        dict
        The optimization problem in optimizer space
        """
        # Create a copy of the problem that is passed in
        optimizer_problem = deepcopy(problem)
        for overall_type in ["variables", "responses"]:
            for element, values in problem[overall_type].items():
                if "bounds" in values:
                    lower = values["bounds"][0]
                    upper = values["bounds"][1]
                    lower = (lower + self.shift[element]) * self.scale[element]
                    upper = (upper + self.shift[element]) * self.scale[element]
                    optimizer_problem[overall_type][element]["bounds"] = [lower, upper]
                    if "default" in values:
                        optimizer_problem[overall_type][element]["default"] = (
                        optimizer_problem[overall_type][element]["default"]
                        + self.shift[element]
                        ) * self.scale[element]
                optimizer_problem[overall_type][element]["shift"] = 0.0
                optimizer_problem[overall_type][element]["scale"] = 1.0
        return optimizer_problem

    def to_design_space_problem(self, problem: dict) -> dict:
        """
        Return the design space problem in Design Space
        This method takes an optimization problem in the optimizer space
        and reverses the shift and scale operations
        Parameters
        ----------
        problem : dict
        The problem in Optimizer Space
        Returns
        -------
        dict
        The design space problem in design space
        """
        # Create a copy of the problem that is passed in
        design_space_problem = deepcopy(problem)
        for overall_type in ["variables", "responses"]:
            for element, values in problem[overall_type].items():
                if "bounds" in values:
                    lower = values["bounds"][0]
                    upper = values["bounds"][1]
                    lower = lower / self.scale[element] - self.shift[element]
                    upper = upper / self.scale[element] - self.shift[element]
                    design_space_problem[overall_type][element]["bounds"] = [lower, upper]
                    if "default" in values:
                        design_space_problem[overall_type][element]["default"] = \
                            design_space_problem[overall_type][element]["default"] \
                                / self.scale[element] - self.shift[element]
                design_space_problem[overall_type][element]["shift"] = self.shift[element]
                design_space_problem[overall_type][element]["scale"] = self.scale[element]
        return design_space_problem
    
    def _dict_to_series(self, var_def: dict) -> Tuple[pd.Series, pd.Series]:
        """Convert shift and scale values for variables/responses in
        optimization problem dictionary to a series.

        :param var_def: Variables/responses dictionary
        :type var_def: dict
        :return: Series containing shift and scale values for each variable/respons
        :rtype: Tuple[Series, Series]
        """
        # Create dictionary to hold shift and scale values
        
        shift = {}
        scale = {}

        # Loop through all variables/responses and get their shift/scale values
        for var, info in var_def.items():
        # Check and assign shift value. Can be any number
            if "shift" not in info or info["shift"] is None:
                shift_val = 0
            else:
                if not isinstance(info["shift"], (int, float, dict, number)):
                    raise TypeError(
                        f"ShiftAndScale: Shift value for {var} must "
                        "be a number or a dict! Received a "
                        f"{type(info['shift']).__name__} instead."
                    )
                if isinstance(info["shift"], dict):
                    if not {"value", "use"}.issubset(set(info["shift"])):
                        raise IndexError(
                            f"ShiftAndScale: Shift value for {var} must "
                            "contain fields for both value and use when"
                            f"a dictionary is used/ Received instead"
                            f"{info['shift']}"
                        )
                    if info["shift"]["use"]:
                        shift_val = info["shift"]["value"]
                    else:
                        shift_val = 0
                else:
                    shift_val = info["shift"]
            shift[var] = shift_val
            # Check and assign shift value. Must be a non-zero number
            if "scale" not in info or info["scale"] is None:
                scale[var] = 1
            else:
                if not isinstance(info["scale"], (int, float, dict, number)):
                    raise TypeError(
                        f"ShiftAndScale: Scale value for {var} must "
                        "be a number or a dict! Received a "
                        f"{type(info['scale']).__name__} instead."
                    )
                if isinstance(info["scale"], dict):
                    if not {"value", "use"}.issubset(set(info["scale"])):
                        raise IndexError(
                            f"ShiftAndScale: Scale value for {var} must "
                            "contain fields for both value and use when"
                            f"a dictionary is used/ Received instead"
                            f"{info['scale']}"
                        )
                    if info["scale"]["use"]:
                        scale_val = info["scale"]["value"]
                    else:
                        scale_val = 1
                else:
                    scale_val = info["scale"]
                if scale_val == 0:
                    raise ValueError(
                        f"ShiftAndScale: Scale value for {var} must "
                        "be a non-zero number!"
                    )
                scale[var] = scale_val
        # Return the values in a series
        return (pd.Series(shift), pd.Series(scale))
        
       
