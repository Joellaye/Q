import numpy as np

def calculate_portfolio_risk_return(returns):
    returns = returns.dropna()
    weights = np.array([1/returns.shape[1]] * returns.shape[1])
    cov_matrix = returns.cov()
    expected_returns = returns.mean()

    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    print("Expected Portfolio Return:", portfolio_return)
    print("Portfolio Variance:", portfolio_variance)
    print("Portfolio Risk (Std Dev):", np.sqrt(portfolio_variance))
