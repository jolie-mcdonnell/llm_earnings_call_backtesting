import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

POSITION_SIZE = 0.65
STOP_LOSS = 0.15
TAKE_PROFIT = 0.50
USE_TRAILING = False
TRAIL_GIVEUP = 0.10

score_columns = [
    "forward_looking_sentiment",
    "management_confidence",
    "risk_and_uncertainty",
    "qa_sentiment",
    "opening_sentiment",
    "financial_performance_sentiment",
    "macroeconomic_reference_sentiment"
]

def backtest_sentiment_strategy(all_calls: pd.DataFrame):
    all_calls = all_calls.fillna(0)
    all_calls['overall_sentiment'] = all_calls[score_columns].mean(axis=1)
    earnings_call_df = all_calls[['date', 'ticker', 'overall_sentiment']]
    earnings_call_df['date'] = earnings_call_df['date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
    
    price_df = yf.download(earnings_call_df.ticker.unique().tolist(),
                           start=earnings_call_df['date'].min() - pd.Timedelta(days=10),
                           end=earnings_call_df['date'].max() + pd.Timedelta(days=10))['Close']
    
    results = {}
    for ticker in earnings_call_df.ticker.unique():
        ec = earnings_call_df[earnings_call_df.ticker == ticker].sort_values('date').copy()
        ec['mu'] = ec['overall_sentiment'].shift().expanding().mean()
        ec['sig'] = ec['overall_sentiment'].shift().expanding().std()
        ec['z_overall'] = (ec['overall_sentiment'] - ec['mu']) / (ec['sig'] + 1e-12)
        upper, lower = 0.75, -0.75
        ec['signal'] = 0
        ec.loc[ec['z_overall'] >= upper, 'signal'] = 1
        ec.loc[ec['z_overall'] <= lower, 'signal'] = -1

        # Map entry date to next trading day
        trading_index = price_df[ticker].index
        ec['entry_date'] = ec['date'].apply(lambda d: trading_index[trading_index.get_indexer([d], method='bfill')[0]])
        
        # Simplified positions & PnL
        pos = pd.Series(0.0, index=trading_index)
        rets = price_df[ticker].pct_change().fillna(0)
        for _, row in ec.iterrows():
            sig = row['signal']
            if sig != 0:
                pos.loc[row['entry_date']:] = sig * POSITION_SIZE
        strategy_daily = (pos * rets).fillna(0.0)
        strategy_curve = (1 + strategy_daily).cumprod()
        results[ticker] = strategy_curve
    return results
