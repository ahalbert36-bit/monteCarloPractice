import numpy as np
from scipy.optimize import minimize


def optimize_portfolio(mean_returns, cov_matrix, weights, target_return=0.0008):
    """Minimize portfolio variance subject to a target return."""
    n_assets = len(mean_returns)

    def objective(portfolio_weights):
        return portfolio_weights @ cov_matrix @ portfolio_weights

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return},
    ]
    bounds = [(0, 1) for _ in range(n_assets)]
    x0 = np.asarray(weights)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x, result.fun
