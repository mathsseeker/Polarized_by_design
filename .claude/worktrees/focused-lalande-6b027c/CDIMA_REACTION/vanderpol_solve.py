from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from vanderpol_function import vanderpol


def solve_vanderpol(
    mu: float,
    x0: float,
    y0: float,
    t0: float = 0.0,
    tf: float = 30.0,
    n: int = 3000,
    method: str = "RK45",
):
    """
    Solve the Van der Pol IVP and return the SciPy solution object.
    """
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, n)

    sol = solve_ivp(
        fun=vanderpol,
        t_span=t_span,
        y0=[x0, y0],
        args=(mu,),
        t_eval=t_eval,
        method=method,
        dense_output=True,
    )
    return sol


def state_at_time(sol, t: float) -> tuple[float, float]:
    """
    Get (x(t), y(t)) from a solution produced with dense_output=True.
    """
    xy = sol.sol(t)
    return float(xy[0]), float(xy[1])


if __name__ == "__main__":
    # Example run (only runs when executing this file directly)
    sol = solve_vanderpol(mu=3, x0=2, y0=-2, tf=30, n=3000)
    x5, y5 = state_at_time(sol, 5.0)
    print(f"State at t=5: x={x5:.2f}, y={y5:.2f}")
