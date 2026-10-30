def vanderpol(t: float, state: tuple[float, float], mu: float) -> tuple[float, float]:
    """
    Van der Pol oscillator (Strogatz form):
        ẋ = μ(y - f(x))
        ẏ = -x/μ
    
    where f(x) = x³/3 - x

    Parameters
    ----------
    t : float
        Time (required by many ODE solvers, even if not used explicitly).
    state : tuple[float, float]
        State vector (x, y).
    mu : float
        Damping parameter (μ > 0).

    Returns
    -------
    (dxdt, dydt) : tuple[float, float]
    
    Reference: Strogatz (2024, chap. 7.5)
    """
    x, y = state
    
    # f(x) = x³/3 - x
    f_x = (x**3) / 3.0 - x
    
    dxdt = mu * (y - f_x)
    dydt = -x / mu
    
    return (dxdt, dydt)
