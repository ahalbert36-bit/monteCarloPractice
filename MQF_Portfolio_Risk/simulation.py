import numpy as np


def build_mean_matrix(mean_returns, num_days):
    """Create a matrix of mean returns with shape (n_stocks, num_days)."""
    mean_values = np.asarray(mean_returns) #converting mean_returns to a column vector of shape (n_stocks, 1)
    mean_returns_matrix = np.tile(mean_values.reshape(-1, 1), (1, num_days)) #repeating mean_values column vector num_days times so each column is identical
    return mean_returns_matrix


def simulate_normal_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims):
    """Run Monte Carlo simulation using a multivariate normal distribution."""
    portfolio_sims = np.full(shape=(num_days, mc_sims), fill_value=0.0) #creates all 0.0 matrix, rows = num_days, columns = mc_sims

    for m in range(mc_sims): #for each column/simulation
        z = np.random.normal(size=(num_days, len(weights))) #array of standard normal vars dimensions (num_days, n_stocks)
        l = np.linalg.cholesky(cov_matrix) #Cholesky decomposition of the covariance matrix comparing stocks to each other
        daily_returns = mean_m + l @ z.T #cholesky matrix*standard normal array + mean returns matrix = daily returns for each stock for each day
        portfolio_sims[:, m] = np.cumprod(daily_returns.T @ weights + 1) * initial_portfolio #for each day, multiply each stock's return by its weight to get the days full portfolio return, repeat for each day

    return portfolio_sims


def simulate_student_t_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims, df=8, rng=None):
    """Run Monte Carlo simulation using a multivariate Student-t distribution."""
    if rng is None: #if no random number generator is provided
        rng = np.random.default_rng() #create default random number generator

    portfolio_sims_t = np.full(shape=(num_days, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        z = np.random.normal(size=(num_days, len(weights)))
        u = rng.chisquare(df, size=num_days).reshape(num_days, 1) #generate chi-square random variables
        l = np.linalg.cholesky(cov_matrix * (df - 2) / df) #cholesky decomposition scaled by (df-2)/df to account for the Student-t distribution's heavier tails
        x = z / np.sqrt(u / df) #z-term adjusted so the covariance is the same as the multivariate normal distribution
        daily_returns = mean_m + l @ x.T
        portfolio_sims_t[:, m] = np.cumprod(daily_returns.T @ weights + 1) * initial_portfolio

    return portfolio_sims_t


def simulate_random_day_model(returns, weights, initial_portfolio, num_days, mc_sims):
    """Sample historical return days with replacement to build a distribution."""
    portfolio_hist = np.full(shape=(num_days, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        sampled_returns = returns.sample(n=num_days, replace=True) #randomly select certain days from historical returns and use their real returns as the returns for each stock
        portfolio_hist[:, m] = np.cumprod(sampled_returns @ weights + 1) * initial_portfolio #randomly selected returns for each stock put together to completely simulate stock returns for each day

    return portfolio_hist
