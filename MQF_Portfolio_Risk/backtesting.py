import numpy as np
import pandas as pd


def run_var_backtest(stock_returns, portfolio_returns, weights, window_backtest=250, alpha_backtest=5, num_sims_backtest=10000, df_backtest=5, rng=None):
    if rng is None:  # if no rng is provided
        rng = np.random.default_rng(10)  # create new rng

    backtest_data = pd.DataFrame(
        index=stock_returns.index[window_backtest:],  # index by date, only uses dates after the initial window_backtest period
        columns=['Normal', 'Student-t', 'Historical'],
        dtype=float,
    )
    backtest_data['Actual Return'] = portfolio_returns.iloc[window_backtest:].to_numpy()  # adds actual portfolio returns for each day in the backtest period

    stocks = stock_returns.columns  # stocks are the same as the column names in the stock-returns DataFrame

    for i in range(window_backtest, len(stock_returns)):  # for each day in the backtest period

        #general info
        historical_window = stock_returns.iloc[i - window_backtest:i]  # historical window of stock returns is 250 days before the current day being evaluated
        mu = historical_window.mean().to_numpy()  # average return for each stock over the historical window
        cov = historical_window.cov().to_numpy()  # covariance matrix of returns for each stock over the historical window

        #normal backtest
        z_normal = rng.standard_normal(size=(num_sims_backtest, len(stocks)))  # standard normal random variables for each stock for each simulation
        L = np.linalg.cholesky(cov)  # Cholesky decomposition of the covariance matrix
        simulated_normal = mu + z_normal @ L.T  # simulate daily returns for each stock
        portfolio_normal = simulated_normal @ weights  # generates full portfolio returns for each day
        backtest_data.iloc[i - window_backtest, 0] = np.percentile(portfolio_normal, alpha_backtest)  # calculates VaR (bottom alpha percentile) out of the simulated portfolio end returns

        #t backtest
        z_t = rng.standard_normal(size=(num_sims_backtest, len(stocks)))  # standard normal random variables for each stock for each simulation
        u_t = rng.chisquare(df_backtest, size=num_sims_backtest)  # chi-square random variables for each simulation
        L_t = np.linalg.cholesky(cov * (df_backtest - 2) / df_backtest)  # cholesky decomposition scaled by (df-2)/df to account for the Student-t distribution's heavier tails
        x_t = z_t / np.sqrt(u_t / df_backtest)[:, None]  # z-term adjusted so the covariance is the same as the multivariate normal distribution
        simulated_t = mu + x_t @ L_t.T
        portfolio_t = simulated_t @ weights  # full-portfolio returns for each day
        backtest_data.iloc[i - window_backtest, 1] = np.percentile(portfolio_t, alpha_backtest)  # VaR of t-test

        #historical backtest
        historical_portfolio = historical_window @ weights  # full portfolio returns for each day
        backtest_data.iloc[i - window_backtest, 2] = np.percentile(historical_portfolio, alpha_backtest)  # VaR of actual data for each day in the backtest period

    for model in ['Normal', 'Student-t', 'Historical']:  # for each model type
        backtest_data[f'{model} Violation (T/F)'] = (backtest_data['Actual Return'] < backtest_data[model])  # if the actual return is less than the VaR for that day, it is a violation

    violation_rates = backtest_data.iloc[:, 4:7].mean().mul(100)
    summary = pd.DataFrame({
        'Model': ['Normal', 'Student-t', 'Historical'],
        'Expected Violations (%)': [5.0, 5.0, 5.0],
        'Actual Violations (%)': violation_rates.values,  # percentage of True's in each column (% of days actual returns < VaR)
    })

    return backtest_data, summary
