import numpy as np
import pandas as pd


def mcValueRisk(returns, alpha=5):
    """Value at Risk: return the percentile for a given confidence level."""
    if isinstance(returns, pd.Series): 
        return np.percentile(returns, alpha) #return the alpha percentile of the returns
    raise TypeError("Expected pandas data series") #raise an error if the input is not a pandas series


def mcConditionalValueRisk(returns, alpha=5):
    """Compute CVaR / expected shortfall below the VaR threshold."""
    if isinstance(returns, pd.Series):
        below_var = returns <= mcValueRisk(returns, alpha=alpha) #generate all returns below alpha percentile of returns
        return returns[below_var].mean() #return mean of all returns below alpha percentile of returns
    raise TypeError("Expected pandas data series")


def summarize_risk(results_df, initial_portfolio, alpha=5):
    """Summarize VaR and CVaR for each model column."""
    VaR = results_df.apply(lambda col: initial_portfolio - mcValueRisk(col, alpha=alpha)) #for each distribution type, determine VaR by subtracting alpha percentile final return from initial portfolio value
    CondVaR = results_df.apply(lambda col: initial_portfolio - mcConditionalValueRisk(col, alpha=alpha)) #for each distribution type, determine CVaR by subtracting mean of returns below alpha percentile from initial portfolio value
    return pd.DataFrame({'VaR': VaR, 'CVaR': CondVaR})
