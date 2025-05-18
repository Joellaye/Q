from src.data_loader import get_data
from src.portfolio import calculate_portfolio_risk_return
from src.regression import run_linear_regression
from src.pca import perform_pca
from src.optimizer import optimize_portfolio


def main():
    data = get_data(['AAPL', 'MSFT', 'GOOGL'], start='2022-01-01', end='2024-01-01')
    calculate_portfolio_risk_return(data)
    run_linear_regression(data)
    perform_pca(data)
    

if __name__ == "__main__":
    main()
