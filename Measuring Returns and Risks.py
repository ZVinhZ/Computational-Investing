# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:37:04 2024

@author: NGUYEN Xuan Vinh
"""
################################################
####### Course : Measuring Returns #######

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

## Calculate and plot annual arithmetic and log returns ##

# Updated portfolio values
values = np.array([10.0 , 12.5 , 8.0 , 13.5 , 7.5 , 15.0])

# Calculating arithmetic and logarithmic returns
arithmetic_returns = ( values [1:] - values [: -1]) / values [: -1]
log_returns = np.log ( values [1:] / values [: -1])

# Plotting the returns
plt.figure ( figsize =(10 , 6))
plt.plot (arithmetic_returns, label = 'Arithmetic Returns', marker ='o')
plt.plot(log_returns, label = 'Logarithmic Returns', marker = 'x')
plt.title('Annual Arithmetic vs. Logarithmic Returns')
plt.xlabel('Year')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()

## Calculate and Plot Cumulative Returns ##

# Calculating cumulative returns
cumulative_arithmetic_return = np.cumsum(arithmetic_returns)
cumulative_logarithmic_return = np.exp(np.cumsum(log_returns)) - 1

# Plotting cumulative returns
plt.figure(figsize = (10, 6))
plt.plot(cumulative_arithmetic_return, label = 'Cumulative Arithmetic Returns', marker ='o')
plt.plot(cumulative_logarithmic_return, label = 'Cumulative Logarithmic Returns', marker ='x')
plt.title ('Cumulative Arithmetic vs. Logarithmic Returns')
plt.xlabel('Year')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

####### Exercise : S&P500 Futures Data Analysis #######
import yfinance as yf
import pandas as pd

# Downloading S&P500 futures data
ticker = 'ES=F' # S&P500 front-month futures ticker symbol
data = yf.download(ticker)

# Creating a pandas dataframe
sp500_data = pd.DataFrame(data)
print(sp500_data.head()) # Displaying the first few row

# Calculate and plot cumulative returns
sp500_price = sp500_data['Adj Close'].to_numpy()
sp500_arithm_ret = (sp500_price[1:] - sp500_price[:-1])/sp500_price[:-1]
sp500_log_ret = np.log(sp500_price[1:]/sp500_price[:-1])
sp500_log_ret = np.log(sp500_data['Adj Close']).diff()

# Calculating cumulative returns
sp500_cumul_arithm_return = np.cumsum(sp500_arithm_ret)
sp500_cumul_log_return = np.exp(np.cumsum(sp500_log_ret)) - 1

# Plot
plt.figure(figsize = (10, 6))
plt.plot(sp500_data.index[1:], sp500_cumul_arithm_return, label = 'Cumulative Arithmetic Returns', marker ='o')
plt.plot(sp500_cumul_log_return, label = 'Cumulative Logarithmic Returns', marker ='x')
plt.title ('S&P 500 Cumulative Arithmetic vs. Logarithmic Returns')
plt.xlabel('Year')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Comment : Over this period, S&P500 has increase for more than 200% 

# Calculate the daily logarithmic returns of the futures prices
sp500_data['Log_Returns'] = np.log(sp500_data['Adj Close'] / sp500_data['Adj Close'].shift(1))

# Annualize the mean of the logaritmic returns
annualized_return = sp500_data['Log_Returns'].mean() * 252
print("annualized_return:", annualized_return)

## Exercise : Visualizing S&P500 Futures Log Returns Distribution
# Determine the number of bins for the histogram
max_daily_return = np.round(sp500_data['Log_Returns'].max(), 2)
min_daily_return = np.round(sp500_data['Log_Returns'].min(), 2)
n_bins = int((max_daily_return - min_daily_return) * 1000)

# Plot a histogram of the logarithmic returns
plt.hist(sp500_data['Log_Returns'], bins = n_bins, color = "orange", label = "Return Distribution")

# Add a vertical dashed line to indicate the mean of the returns
plt.axvline(sp500_data['Log_Returns'].mean(), color = 'black', linestyle = 'dashed', linewidth = 2, label = 'Daily Mean Return')

# Configure the plot
plt.title('Histogram of S&P500 Futures Logarithmic Returns')
plt.xlabel('Logarithmic Returns')
plt.ylable('Frequency(Log)')
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) # format y-axis labels as integers
plt.legend()

# Display the plot
plt.show()

################################################
####### Course : Measuring Risk #######

####### Exercise : Calculating Annualized Volatility #######
# Calculate the sd (vol) of the log returns
volatility = sp500_data['Log_Returns'].std()

# Annualize the volatility
# There are approximately 252 trading days in a year
annualized_volatility = volatility * np.sqrt(252)
print("Annualized Volatility:", annualized_volatility)


####### Exercise : Analyzing S&P500 Log Returns distribution with SD lines #######
# mean of log returns
mean_return = sp500_data['Log_Returns'].mean()
std_return = volatility


# Determine the number of bins for the histogram
max_daily_return = np.round(sp500_data['Log_Returns'].max(), 2)
min_daily_return = np.round(sp500_data['Log_Returns'].min(), 2)
n_bins = int((max_daily_return - min_daily_return) * 100)

# Plot a histogram of the logarithmic returns
plt.hist(sp500_data['Log_Returns'], bins = n_bins, color = "orange", label = "Return Distribution")

# Adding vertical lines for mean and sd
plt.axvline(mean_return, color = 'black', linestyle = 'dashed', linewidth = 2, label = 'Mean Return')

for i in range(1, 4):
    plt.axvline(mean_return + i * std_return, color = 'green', linestyle = 'dashed', linewidth = 1, label = f"+{i} STD")
    plt.axvline(mean_return - i * std_return, color = 'green', linestyle = 'dashed', linewidth = 1, label = f"+{i} STD")
    
####### Exercise : Comparing S&P500 log ret with a normal distribution #######
mean, std = mean_return, std_return

log_returns = sp500_data['Log_Returns'].dropna()

# Generate range of values for Normal distribution
n_bins = int(max(log_returns) - min(log_returns) * 10000)
norm_dist = np.linspace(min(log_returns), max(log_returns), n_bins)

# Calculate the normal distribution with the same mean and standard deviation
normal_curve = norm.pdf(norm_dist, mean, std)

# Plot a histogram of the logarithmic returns
plt.figure(figsize=(10, 6))
plt.hist(log_returns, bins=n_bins, color="orange", density=True, label="Return Distribution")
plt.plot(norm_dist, normal_curve, label="Normal Distribution", color="blue")
plt.title("Logarithmic Returns Distribution and Normal Distribution")
plt.xlabel("Logarithmic Returns")
plt.ylabel("Frequency")
plt.legend()
plt.show()


####### Exercise : Calculating and Plotting Ongoing Drawdown of S&P500 #######
# Calculate cumulative returns
sp500_data['Cumulative_Returns'] = sp500_data['Log_Returns'].cumsum()

# Caculate ongoing drawdown
rolling_max = sp500_data['Cumulative_Returns'].cummax()
sp500_data['Drawdown'] = rolling_max - sp500_data['Cumulative_Returns']

# Plotting the results
fig, ax = plt.subplots()
ax.fill_between(sp500_data.index, sp500_data['Drawdown'], color = 'red', alpha = 0.3)
ax.plot(sp500_data['Cumulative_Returns'], label = 'Cumulative_Returns')
ax.set_title('Cumulative Returns and Ongoing Drawdown')
ax.set_xlabel('Date')
ax.set_ylable('Returns/Drawdown')
ax.legend()
plt.show()

####### Exercise : Calculating and Plotting Ongoing Drawdown of a Mixed Portfolio #######
# S&P500 and US 10-y Treasury futures data
sp500_futures = yf.download('ES=F')['Adj Close']
treasury_futures = yf.download('ZN=F')['Adj Close']

# Align the datasets
data = pd.DataFrame({'SP500': sp500_futures,
                     'Treasury': treasury_futures}).dropna()

# Calculate daily logarithmic returns
data['SP500_Returns'] = np.log(data['SP500']/data['SP500'].shift(1))
data['Treasury_Returns'] = np.log(data['Treasury']/data['Treasury'].shift(1))

# Calculate portfolio returns (60% SP500, 40% Treasury)
data['Portfolio_Returns'] = 0.6 * data['SP500_Returns'] + 0.4 * data['Treasury_Returns']

# Calculate cumulative returns for SP500, Treasury, and Portfolio
data['Cumulative_SP500_Returns'] = data['SP500_Returns'].cumsum()
data['Cumulative_Treasury_Returns'] = data['Treasury_Returns'].cumsum()
data['Cumulative_Returns'] = data['Portfolio_Returns'].cumsum()

# Calculate ongoing drawdown
rolling_max = data['Cumulative_Returns'].cummax()
data['Drawdown'] = rolling_max - data['Cumulative_Returns']

# Plotting
fig, ax = plt.subplots()

ax.fill_between(data.index, data['Drawdown'], color = 'red', alpha = 0.3)
ax.plot(data['Cumulative_Returns'], label = 'Portfolio Cumulative Returns')
ax.plot(data['Cumulative_SP500_Returns'], label = 'S&P500 Cumulative Returns', linestyle = '--', linewidth = 0.5)
ax.plot(data['Cumulative_Treasury_Returns'], label = '10-Year Treasury Cumulative Returns', linestyle = '--', linewidth = 0.5)
ax.set_title('Portfolio Cumulative Returns and Ongoing Drawdown')
ax.set_xlabel('Date')
ax.set_ylabel('Returns/Drawdown')
ax.legend()
plt.show()

print("Maximum Drawdown:", data['Drawdown'].max())













