# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:19:46 2024

@author: NGUYEN Xuan Vinh
"""

################################################
####### Course : Measuring Risk-Adjusted Returns #######

##### 1 - Sharpe Ratio ########################

##### Exercise : Calculating Sharpe Ratio of SP500 futures with annualized data #####
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Download SP500 front month futures data
sp500_futures = yf.download('ES=F')['Adj Close']

# Calculate daily log returns
sp500_futures['Log_Returns'] = np.log(sp500_futures/sp500_futures.shift(1))

# Annualize Returns and Volatility
annualized_return = sp500_futures['Log_Returns'].mean()*252
annualized_volatility = sp500_futures['Log_Returns'].std()*np.sqrt(252)

# Download the 3-month Treasury bill rate as the risk-free rate
sp500_start_date = str(sp500_futures.index[0])[:10]
risk_free_rate_series = yf.download('^IRX', start = sp500_start_date)['Adj Close']
risk_free_rate = risk_free_rate_series.mean()/100

# Calculate the annual Sharp Ratio
sharp_ratio = (annualized_return - risk_free_rate) / annualized_volatility


# another way
risk_free = risk_free_rate_series/100/252
adjusted_return = sp500_futures['Log_Returns'] - risk_free
mean_adjusted_return = adjusted_return.mean()*252
annualized_sharp_ratio = mean_adjusted_return/annualized_volatility

print(annualized_sharp_ratio)
print(pd.DataFrame([sp500_futures['Log_Returns'], risk_free, adjusted_return]).transpose())


print("annualized_return:", annualized_return)
print("risk_free_rate_annual:", risk_free_rate)
print("annualized_volatility:", annualized_volatility)
print("sharp_ratio:", sharp_ratio)

########################
##### Exercise : Calculating Sharpe Ratio of 10-Year Treasury Futures with annualized data #####

# Download 10Y treasury Futures data
T_futures = yf.download('ZN=F', start = sp500_start_date)['Adj Close']

# Calculate daily log returns
T_futures['Log_Returns'] = np.log(T_futures/T_futures.shift(1))

# Annualize Returns and Volatility
T_annualized_return = T_futures['Log_Returns'].mean()*252
T_annualized_volatility = T_futures['Log_Returns'].std()*np.sqrt(252)

# Download the 3-month Treasury bill rate as the risk-free rate
risk_free_rate_series = yf.download('^IRX', start = sp500_start_date)['Adj Close']
risk_free_rate = risk_free_rate_series.mean()/100

# Calculate the annual Sharp Ratio
sharp_ratio = (T_annualized_return - risk_free_rate) / T_annualized_volatility

print("annualized_return:", T_annualized_return)
print("risk_free_rate_annual:", risk_free_rate)
print("annualized_volatility:", T_annualized_volatility)
print("sharp_ratio:", sharp_ratio)

# Sharp ratio is lower than SP500 because 10Y Treasury bill is less risky

########################
##### Exercise : Calculating Sharpe Ratio of a Mixed portfolio with SP500 and 10-Year Treasury Futures with annualized data #####
mixed_log_returns = 60/100 * sp500_futures['Log_Returns'] + 40/100 * T_futures['Log_Returns']
mixed = pd.DataFrame()

mixed['Log_Returns'] = mixed_log_returns


# Annualized Returns and Volatility of mixed portfolio
mixed_annualized_return = mixed_log_returns.mean()*252
mixed_annualized_volatility = mixed_log_returns.std()*np.sqrt(252)

# Sharp ratio
mixed_sharp_ratio = (mixed_annualized_return - risk_free_rate) / mixed_annualized_volatility

# Calculate cumulative returns
mixed['Cumulative_Returns'] = mixed_log_returns.cumsum()

# Caculate ongoing drawdown
rolling_max = mixed['Cumulative_Returns'].cummax()
mixed['Drawdown'] = rolling_max - mixed['Cumulative_Returns']
mixed_skew = mixed['Log_Returns'].skew()
mixed_kurtosis = mixed['Log_Returns'].kurtosis()


print('annualized returns:', mixed_annualized_return)
print('annualized volatility:', mixed_annualized_volatility)
print('skew:', mixed_skew)
print('kurtosis:', mixed_kurtosis)
print('max_drawdown:', mixed['Drawdown'].max())


# Plot the cumul returns and drawdowns
sp500_futures['Cumulative_Returns'] = sp500_futures['Log_Returns'].cumsum()
T_futures['Cumulative_Returns'] = T_futures['Log_Returns'].cumsum()
plt.plot(mixed['Cumulative_Returns'], label='Mixed Portfolio')
plt.plot(sp500_futures['Cumulative_Returns'], label='SP500', linestyle = 'dashed')
plt.plot(T_futures['Cumulative_Returns'], label='10 Year Treasury', linestyle = 'dashed')


plt.plot(mixed['Drawdown'], label = 'Drawdown')
plt.title('Cumul Returns and Drawdown')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()


################################################
##### 2 - Inflation-Adjusted Sharp Ratio ########################

########################
##### Exercise : Calculating Inflation-Adjusted Sharpe Ratio of a Mixed portfolio with SP500 and 10-Year Treasury Futures with annualized data #####
import pandas_datareader as pdr

# Specify the data source as 'fred' (Federal Reserve Economic Data)
data_source = 'fred'

# Specify the series ID for the Consumer Price Index for All Urban Consumers (CPIAUCSL)
series_id = 'CPIAUCSL'

# Define the start and end dates for the data
start_date = sp500_start_date


# fetch the inflation data
inflation_data = pdr.get_data_fred(series_id, start_date)
# Calculate the annualized inflation rate
inflation_rate = inflation_data['CPIAUCSL'].pct_change(periods = 12)
# Resample monthly inflation data to daily frequency
inflation_rate = inflation_rate.resample('D').ffill()

# inflation_adjusted Sharp ratio
inflation_Sharp_ratio = (mixed_annualized_return - inflation_rate.mean())/mixed_annualized_volatility
print(inflation_Sharp_ratio)

########################
##### Exercise : Calculating Inflation-Adjusted Sharpe Ratio of a Mixed portfolio  #####

# Download Asset Data
US_Eqt = yf.download('SPY', start = sp500_start_date)['Adj Close']
Foreign_Developed_Eqt = yf.download('EFA', start = sp500_start_date)['Adj Close']
Emerg_Market_Eqt = yf.download('EEM', start = sp500_start_date)['Adj Close']
US_REIT = yf.download('VNQ', start = sp500_start_date)['Adj Close']
US_Treasury_Bond = yf.download('IEF', start = sp500_start_date)['Adj Close']
US_TIPS = yf.download('TIP', start = sp500_start_date)['Adj Close']

# Calculate daily log return
US_Eqt['Log_Returns'] = np.log(US_Eqt/US_Eqt.shift(1))
Foreign_Developed_Eqt['Log_Returns'] = np.log(Foreign_Developed_Eqt/Foreign_Developed_Eqt.shift(1))
Emerg_Market_Eqt['Log_Returns'] = np.log(Emerg_Market_Eqt/Emerg_Market_Eqt.shift(1))
US_REIT['Log_Returns'] = np.log(US_REIT/US_REIT.shift(1))
US_Treasury_Bond['Log_Returns'] = np.log(US_Treasury_Bond/US_Treasury_Bond.shift(1))
US_TIPS['Log_Returns'] = np.log(US_TIPS/US_TIPS.shift(1))

# Mixed Portfolio
mixed_portfolio_return = 0.3 * US_Eqt['Log_Returns'] + 0.15 * Foreign_Developed_Eqt['Log_Returns'] + 0.05 * Emerg_Market_Eqt['Log_Returns'] + 0.2 * US_REIT['Log_Returns'] + 0.15 * US_Treasury_Bond['Log_Returns'] + 0.15 *US_TIPS['Log_Returns']
mixed_portfolio_return.dropna(inplace = True)
annualized_mixed_portfolio_return = mixed_portfolio_return.mean()*252
annualized_mixed_portfolio_vol = mixed_portfolio_return.std()*np.sqrt(252)

# inflation_adjusted Sharp ratio
inflation_Sharp_ratio_portfolio = (annualized_mixed_portfolio_return- inflation_rate.mean())/mixed_annualized_volatility
print(inflation_Sharp_ratio_portfolio)

################################################
##### 3 - Information Ratio ########################

# Tracking erro = the standard deviation of the excess return of a portfolio over its benchmark
# Key measure in portfolio management
# It quantifies the consistency and magnitude of the portfolio's performance relative to the benchmark.
# Interpretation of Tracking error
# - A higher tracking error indicates a greater divergence from the benchmark, signyfying higher active risk
# - A lower tracking erro implies that the portfolio closely follows its benchmark

########################
##### Exercise : Calculating IR of Berkshire Hathaway with SP500 as benchmark

# Download Berkshire Hathaway stock data
berkshire = yf.download('BRK-A')['Adj Close']
sp500_futures = yf.download('ES=F')['Adj Close']

# Compute latest starting date
latest_start_date = max(berkshire.index[0], sp500_futures.index[0])

# Use only data starting from the latest starting data, so that we have the same data points for both series
berkshire = berkshire.loc[latest_start_date:]
sp500_futures = sp500_futures.loc[latest_start_date:]

# Calculate daily log returns
berkshire['Log_Returns'] = np.log(berkshire/berkshire.shift(1))
sp500_futures['Log_Returns'] = np.log(sp500_futures/sp500_futures.shift(1))

# Compute excess returns
excess_returns = berkshire['Log_Returns'] - sp500_futures['Log_Returns']

# Annualize Excess Returns and Calculate IR
annualized_excess_return = excess_returns.mean() * 252
tracking_error = excess_returns.std() * np.sqrt(252)

# Calculate the IR
information_ratio = annualized_excess_return / tracking_error
print('Information Ratio:', information_ratio)

########################
##### Exercise : Calculating IR of equally-weighted FAANG Portfolio with NASDAQ as Benchmark ######
# Download stock data for FAANG companies
faang_tickers = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL']
faang_data = yf.download(faang_tickers, start = '2010-01-01')['Adj Close']
faang_data.dropna(inplace = True)
# Download NASDAQ index data

nasdaq_data = yf.download('NQ=F')['Adj Close']
nasdaq_data = nasdaq_data.loc[faang_data.index[0]:]

# Daily log returns
faang_data['META_log_returns'] = np.log(faang_data['META']/faang_data['META'].shift(1))
faang_data['AAPL_log_returns'] = np.log(faang_data['AAPL']/faang_data['AAPL'].shift(1))
faang_data['AMZN_log_returns'] = np.log(faang_data['AMZN']/faang_data['AMZN'].shift(1))
faang_data['NFLX_log_returns'] = np.log(faang_data['NFLX']/faang_data['NFLX'].shift(1))
faang_data['GOOGL_log_returns'] = np.log(faang_data['GOOGL']/faang_data['GOOGL'].shift(1))


nasdaq_data['Log_Returns'] = np.log(nasdaq_data/nasdaq_data.shift(1))

faang_data['EW_returns'] = (faang_data['META_log_returns'] + faang_data['AAPL_log_returns'] + faang_data['AMZN_log_returns'] + faang_data['NFLX_log_returns'] + faang_data['GOOGL_log_returns'])/5


# Excess returns
excess_returns_faang = faang_data['EW_returns'] - nasdaq_data['Log_Returns']
# annualized excess returns
annualized_excess_return_faang = excess_returns_faang.mean()*252
tracking_error_faang = excess_returns_faang.std() * np.sqrt(252)
# IR faang
IR_faang = annualized_excess_return_faang / tracking_error_faang

print('Information Ratio:', IR_faang)


################################################
##### 4 - Sortino Ratio ########################


################################################
##### 5 - Calmar Ratio ########################












