def does_sum_up_to(a: float, b: float, to: float, epsilon=1e-7) -> bool:
    """
    Check if the sum of two numbers, `a` and `b`, is approximately equal to a target value `to` within a given `epsilon`.

    ---
    ### Parameters:
        - `a` (`float`): The first number.
        - `b` (`float`): The second number.
        - `to` (`float`): The target sum value.
        - `epsilon` (`float`, optional): The acceptable margin of error. Defaults to `1e-7`.

    ---
    ### Returns:
        - `bool`: `True` if the sum of `a` and `b` is within `epsilon` of the target value `to`, `False` otherwise.
    """
    return abs(a + b - to) < epsilon
