import numpy as np
from scipy.optimize import minimize


def equal_weight_portfolio(num_assets):
    """Return a fully diversified equal-weight portfolio."""
    return np.full(num_assets, 1 / num_assets)


def optimize_portfolio(mean_returns, cov_matrix, target_return=0.0008):
    """Minimize portfolio variance subject to a target return."""
    n_assets = len(mean_returns)

    def objective(weights):
        return weights @ cov_matrix @ weights

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return},
    ]
    bounds = [(0, 1) for _ in range(n_assets)]
    x0 = equal_weight_portfolio(n_assets)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x, result.fun
