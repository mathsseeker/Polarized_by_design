def cdima(t: float, state: tuple[float, float], a: int, b: int) -> tuple[float, float]:
    """
    CDIMA model:
        x' = a - x - (4*x*y)/(1 + x^2)
        y' = b*x*(1 - y/(1 + x^2))

    Parameters
    ----------
    t : float
        Time (required by many ODE solvers, even if not used explicitly).
    state : tuple[float, float]
        State vector.
    a, b : float
        Positive parameters.

    Returns
    -------
    (dxdt, dydt) : tuple[float, float]
    """
    x, y = state
    denom = 1.0 + x * x

    dxdt = a - x - (4.0 * x * y) / denom
    dydt = b * x * (1.0 - y / denom)

    return (dxdt, dydt) 

