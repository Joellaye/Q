from sklearn.linear_model import LinearRegression

def run_linear_regression(returns):
    X = returns[['MSFT', 'GOOGL']].values
    y = returns['AAPL'].values

    model = LinearRegression()
    model.fit(X, y)

    print("Linear Regression Coefficients (Betas):", model.coef_)
    print("Intercept:", model.intercept_)
