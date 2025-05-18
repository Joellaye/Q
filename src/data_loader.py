import yfinance as yf

def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.pct_change().dropna()
