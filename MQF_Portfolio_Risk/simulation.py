import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_mean_matrix(mean_returns, num_days):
    mean_values = np.asarray(mean_returns) #converting mean_returns to a column vector of shape (n_stocks, 1)
    mean_returns_matrix = np.tile(mean_values.reshape(-1, 1), (1, num_days)) #repeating mean_values column vector num_days times so each column is identical
    return mean_returns_matrix


def simulate_normal_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims, rng=None):
    if rng is None: #if no random number generator is provided
        rng = np.random.default_rng(10) #create default random number generator

    portfolio_sims = np.full(shape=(num_days, mc_sims), fill_value=0.0) #creates all 0.0 matrix, rows = num_days, columns = mc_sims

    l = np.linalg.cholesky(cov_matrix) #Cholesky decomposition of the covariance matrix comparing stocks to each other

    for m in range(mc_sims): #for each column/simulation
        z = rng.standard_normal(size=(num_days, len(weights))) #array of standard normal vars dimensions (num_days, n_stocks)
        daily_returns = mean_m + l @ z.T #cholesky matrix*standard normal array + mean returns matrix = daily returns for each stock for each day
        portfolio_sims[:, m] = np.cumprod(daily_returns.T @ weights + 1) * initial_portfolio #for each day, multiply each stock's return by its weight to get the days full portfolio return, repeat for each day

    return portfolio_sims


def simulate_student_t_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims, df=5, rng=None):
    if rng is None: #if no random number generator is provided
        rng = np.random.default_rng(10) #create default random number generator

    portfolio_sims_t = np.full(shape=(num_days, mc_sims), fill_value=0.0)

    l = np.linalg.cholesky(cov_matrix * (df - 2) / df) #cholesky decomposition scaled by (df-2)/df to account for the Student-t distribution's heavier tails

    for m in range(mc_sims):
        z = rng.standard_normal(size=(num_days, len(weights)))
        u = rng.chisquare(df, size=num_days).reshape(num_days, 1) #generate chi-square random variables
        x = z / np.sqrt(u / df) #z-term adjusted so the covariance is the same as the multivariate normal distribution
        daily_returns = mean_m + l @ x.T
        portfolio_sims_t[:, m] = np.cumprod(daily_returns.T @ weights + 1) * initial_portfolio

    return portfolio_sims_t


def simulate_random_day_model(historical_stock_returns, weights, initial_portfolio, num_days, mc_sims, rng=None):
    if rng is None: #if no random number generator is provided
        rng = np.random.default_rng(10) #create default random number generator

    portfolio_hist = np.full(shape=(num_days, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        sample_indices = rng.integers(0, len(historical_stock_returns), size=num_days) #selects random row positions
        sampled_returns = historical_stock_returns.iloc[sample_indices] #selects the data from random row positions to make new dataframe
        portfolio_hist[:, m] = np.cumprod(sampled_returns @ weights + 1) * initial_portfolio #randomly selected returns for each stock put together to completely simulate stock returns for each day

    return portfolio_hist


def plot_sample_paths(sims, title, color, initial_portfolio): #displays 25 random simulations per model

    sim_number = np.random.default_rng(10).choice(sims.shape[1], size=25, replace=False) #randomly selects 25 unique simulations by index number

    for col in sim_number: #for each simulation
        plt.plot(sims[:, col], color=color, alpha=0.35) #line plot of data from all dates in each selected sim_number 

    plt.axhline(initial_portfolio, color='black', linestyle='--', label='Initial Portfolio') #horizontal line to show starting portfolio value
    plt.title(title)
    plt.xlabel('Trading Day')
    plt.xlim(0,250)
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.show()


def plot_final_value_distributions(normal_sims, t_sims, hist_sims, risk_summary, initial_portfolio): #final portfolio values, VaR and CVaR as inputs

    sims = [normal_sims, t_sims, hist_sims] #df of data from all simulations, all model types
    labels = ['normal', 't', 'hist']
    titles = ['Normal', 'Student-t', 'Historical']

    for i in range(3): #for each model
        plt.subplot(1, 3, i+1) #3 subplots, 1 row, 3 cols, position changes for each model type

        vals = sims[i][-1, :] #model type i, final trading day, every simulation

        plt.hist(vals, bins=50) #plot final values into 50 bins
        plt.axvline(initial_portfolio - risk_summary.loc[labels[i], 'VaR'], color='red', label='VaR') #vertical line of initial value - VaR
        plt.axvline(initial_portfolio - risk_summary.loc[labels[i], 'CVaR'], color='orange', label='CVaR')

        plt.title(titles[i]) #title df
        plt.xlabel('Portfolio Value ($)')
        plt.legend()

    plt.ylabel('Frequency')
    plt.show()


def plot_var_cvar_comparison(risk_summary): #comparing VaR and CVar by model

    x = np.arange(len(risk_summary)) #number of models

    plt.bar(x - 0.2, risk_summary['VaR'], 0.4, label='VaR') #VaR bar slightly to left, VaR data
    plt.bar(x + 0.2, risk_summary['CVaR'], 0.4, label='CVaR')

    plt.xticks(x, risk_summary.index) #x-ticks with model names
    plt.ylabel('Dollar Loss ($)')
    plt.title('VaR & CVaR by Model')
    plt.legend()
    plt.show()


def plot_violation_rates(backtest_summary): #plot backtest summary data
    backtest_summary.plot.bar(x='Model', y=['Expected Violations (%)', 'Actual Violations (%)'], figsize=(10, 8), color=['lightblue', 'salmon'])
    plt.axhline(y=5, linestyle='--', label='Expected Violation Rate (5%)') #horizontal line
    plt.title('Expected vs Actual Violation Rates by Model')
    plt.ylabel('Violation Rate (%)')
    plt.ylim(0, 6)
    plt.legend(loc='upper left')
    plt.show()
