import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def build_window_dates(days_back=365):
    end_date = dt.datetime(2026, 9, 30) #9/30/2026
    start_date = end_date - dt.timedelta(days=days_back)
    return start_date, end_date


def get_data(stocks, start, end, trade_days=250):
    stock_data = yf.download(stocks, start=start, end=end, auto_adjust=True,progress=False) #download stock data from Yahoo Finance

    if isinstance(stock_data.columns, pd.MultiIndex): #if stock data has multiple columns
        stock_data = stock_data['Close'] #select column labeled 'Close'
    else: #if stock data has 1 column
        if 'Close' in stock_data.columns: #if the 1 column is labeled 'Close'
            stock_data = stock_data[['Close']] #select the column labeled 'Close'
        else: #if the 1 column is not labeled 'Close'
            raise ValueError("The downloaded data does not contain a 'Close' column.") #raise an error
    stock_data = stock_data.reindex(columns=stocks)  # reorder columns to match stocks list
    #stock data is now only closing values for each day in 250-day span

    returns = stock_data.pct_change().tail(trade_days) #compute percentage change in closing value for each day in 250-day span
    mean_returns = returns.mean() #mean of daily returns for the stock
    cov_matrix = returns.cov() #covariance matrix of daily returns for the stock



    return mean_returns, cov_matrix, returns


def plot_stock_prices(stock_returns, weights, initial_portfolio): #stock values over time

    weights = np.asarray(weights) #weights in array form
    initial_values = initial_portfolio * weights #initial investment value for each stock

    stock_values = initial_values * (1 + stock_returns).cumprod() #value of each stock over time based on returns

    for col in stock_values.columns: #for each stock
        plt.plot(stock_values.index, stock_values[col], label=col) #x axis of date, y axis of investment value, label of stock

    plt.title('Individual Stock Investment Values')
    plt.xlabel('Date')
    plt.ylabel('Investment Value ($)')
    plt.legend()
    plt.show()


def get_historical_portfolio_returns(stocks, weights, years_back=5):
    end_date = pd.Timestamp("2026-09-30").normalize() #fixed end date: 9/30/2026
    start_date = end_date - pd.DateOffset(years=years_back) #date 5 years before the fixed end date

    prices = yf.download(stocks,start=start_date,end=end_date,auto_adjust=True,progress=False)['Close'] #closing prices for each stock on each day in 5-year span
    prices = prices.reindex(columns=stocks)  # reorder columns to match stocks list

    returns = prices.pct_change().dropna() #percentage change in closing value for each stock and for each day in 5-year span
    portfolio_returns = returns @ weights #multiplies the returns of each stock by its weight, sums returns for each day to get total portfolio return for each day in 5-year span
    return returns, portfolio_returns
