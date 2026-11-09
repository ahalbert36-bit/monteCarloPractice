import pandas as pd

from data import build_window_dates, get_data, get_historical_portfolio_returns
from simulation import (
    build_mean_matrix,
    simulate_normal_model,
    simulate_random_day_model,
    simulate_student_t_model,
)
from risk import summarize_risk
from backtesting import run_var_backtest


def main():
    stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    weights = [0.25, 0.25, 0.25, 0.25]
    initial_portfolio = 1000 #starting amount of money
    mc_sims = 500 #number of simulations
    num_days = 50 #number of days being measured
    alpha = 5 #bottom 5% being evaluated for risk

    start_date, end_date = build_window_dates(days_back=300)
    mean_returns, cov_matrix, returns = get_data(stocks, start_date, end_date) #returning values for chosen stocks over the last 300 days
    mean_m = build_mean_matrix(mean_returns, num_days) #mean_returns column vector repeated num_days times

    normal_sims = simulate_normal_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims) #simulates full portfolio returns for each day
    t_sims = simulate_student_t_model(mean_m, cov_matrix, weights, initial_portfolio, num_days, mc_sims, df=8)
    hist_sims = simulate_random_day_model(returns, weights, initial_portfolio, num_days, mc_sims)

    results_df = pd.DataFrame({ #creates dataframe with the final portfolio value for each simulation for each model
        'normal': normal_sims[-1, :],
        't': t_sims[-1, :],
        'hist': hist_sims[-1, :],
    })

    risk_summary = summarize_risk(results_df, initial_portfolio, alpha=alpha) #generate VaR and CVaR for each model based on final portfolio returns
    print('Risk Summary')
    print(risk_summary)

    returns_past_data, portfolio_returns_past_data = get_historical_portfolio_returns(stocks, weights, years_back=5)
    backtest_data, summary = run_var_backtest(
        returns_past_data,
        portfolio_returns_past_data,
        weights,
        window_backtest=250, #number of days to use for historical data to calculate VaR
        alpha_backtest=0.05,
        num_sims_backtest=10000,
        df_backtest=8, #same df as used in the Student-t model
    )

    print('\nVaR Backtest Summary')
    print(summary)


if __name__ == '__main__': #if the file is being run directly
    main()
