import pandas as pd
import numpy as np 
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.title("🎲🎰 Probability Distribution Calculator")

# Sidebar
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
threshold = st.sidebar.number_input("Enter Threshold (%)", value=5.0, step=0.1)

# Download data
data = yf.download(ticker, start=start_date, end=end_date)

# Display raw data

weekly_data = data.resample('W').agg({('Close', ticker): ['first', 'last']})
weekly_data.columns = ['start_of_week_price', 'end_of_week_price']
weekly_data['start_of_week'] = weekly_data.index - pd.offsets.Week(weekday=0) + pd.offsets.BDay()
weekly_data['end_of_week'] = weekly_data.index - pd.offsets.Week(weekday=4) + pd.offsets.BDay()
weekly_data['PriceChange'] = weekly_data['end_of_week_price'] - weekly_data['start_of_week_price']
weekly_data['percent_change'] = (weekly_data['PriceChange'] / weekly_data['start_of_week_price']) * 100
weekly_data
st.subheader(f"{ticker} Weekly Data")
st.write(weekly_data.tail())


#Calculating the weekly price change and percent change

percent_change = weekly_data['percent_change'].dropna()

#plot the histogram
plt.figure(figsize=(10, 6))
sns.histplot(percent_change, bins=30, kde=True, stat='density')

(mu, sigma) = norm.fit(percent_change)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.axvline(mu, color='red', linestyle='--', label='Mean')
plt.axvline(mu + sigma, color='Green', linestyle='--', label='1 Std Dev')
plt.axvline(mu - sigma, color='Green', linestyle='--', label='-1 Std Dev')
plt.axvline(mu + 2*sigma, color='blue', linestyle='--', label='2 Std Dev')
plt.axvline(mu - 2*sigma, color='blue', linestyle='--', label='-2 Std Dev')


st.pyplot(plt)
st.subheader(f"{ticker} Weekly Price Change Distribution")
st.write("This histogram shows the distribution of weekly price changes for the selected stock. The red line indicates the mean, while the green and blue lines indicate one and two standard deviations from the mean, respectively.")

# Calculate and display returns
st.subheader(f"{ticker} Return Distribution")

probability = norm.sf(threshold, loc=mu, scale=sigma)
st.write(f"Probability of {ticker} stock going up more than {threshold}% in one week is {round(probability*100,1):.1f}%")
