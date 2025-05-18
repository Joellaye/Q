import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(returns, risk_free_rate=0.0):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    def portfolio_performance(weights):
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_std
        return -sharpe  # Negative because we minimize

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # Bounds: weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    # Initial guess: equal weights
    init_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_performance, init_guess, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    print("Optimized Weights:", result.x)
    print("Max Sharpe Ratio:", -result.fun)
    return result.x
