import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

# Strategy Parameters
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
    """
    This function forms a strategy around each ticker's deviation in overall earnings call sentiment score over time
    The ticker's price data is scraped to serve as a comparison between the sentiment strategy and buy/hold

    Input:
    all_calls (pd.DataFrame): dataframe of all earnings calls sentiment data

    Output:
    results (pd.DataFrame): dataframe of returns for each ticker and strategy
    """

    all_calls = all_calls.fillna(0)
    all_calls['overall_sentiment'] = all_calls[score_columns].mean(axis=1)
    earnings_call_df = all_calls[['date', 'ticker', 'overall_sentiment']]
    earnings_call_df['date'] = pd.to_datetime(earnings_call_df['date'])

    tickers = earnings_call_df['ticker'].unique().tolist()
    start = earnings_call_df['date'].min() - pd.Timedelta(days=10)
    end   = earnings_call_df['date'].max() + pd.Timedelta(days=10)

    price_df = yf.download(tickers, start=start, end=end)['Close']

    results = {}

    for ticker in tickers:
        px = price_df[ticker].dropna()
        rets = px.pct_change().fillna(0.0)
        trading_index = px.index

        ec = earnings_call_df[earnings_call_df.ticker == ticker].sort_values('date').copy()
        ec['mu'] = ec['overall_sentiment'].shift().expanding().mean()
        ec['sig'] = ec['overall_sentiment'].shift().expanding().std()
        ec['z_overall'] = (ec['overall_sentiment'] - ec['mu']) / (ec['sig'] + 1e-12)

        # Signal thresholds
        upper, lower = 0.75, -0.75
        ec['signal'] = 0
        ec.loc[ec['z_overall'] >= upper, 'signal'] = 1
        ec.loc[ec['z_overall'] <= lower, 'signal'] = -1

        # Map to next trading day
        def next_trading_day(d):
            if d in trading_index:
                i = trading_index.get_loc(d)
                return trading_index[min(i + 1, len(trading_index) - 1)]
            else:
                i = trading_index.get_indexer([d], method='bfill')[0]
                return trading_index[i]

        ec['entry_date'] = ec['date'].apply(next_trading_day)
        ec = ec.dropna(subset=['entry_date'])

        # Initialize position and tracking
        pos = pd.Series(0.0, index=trading_index)
        trade_entries, trade_exits, trade_pnls = [], [], []

        for i, row in ec.iterrows():
            sig = row['signal']
            if sig == 0:
                continue

            entry = row['entry_date']
            # Find next earnings call
            nxt = ec.loc[ec['entry_date'] > entry, 'entry_date'].min()
            last = trading_index[-1]
            exit_plan = nxt if pd.notna(nxt) else last

            # Defensive alignment
            if entry not in trading_index:
                entry = trading_index[trading_index.get_indexer([entry], method='bfill')[0]]
            if exit_plan not in trading_index:
                exit_plan = trading_index[trading_index.get_indexer([exit_plan], method='bfill')[0]]

            entry_price = px.loc[entry]
            best_fav = 0.0
            actual_exit = exit_plan

            window = pos.loc[entry:exit_plan].index
            for d in window:
                pnl = sig * (px.loc[d] / entry_price - 1.0)
                if pnl > best_fav:
                    best_fav = pnl

                stopped = False
                if pnl <= -STOP_LOSS:
                    actual_exit = d; stopped = True
                elif pnl >= TAKE_PROFIT:
                    actual_exit = d; stopped = True
                elif USE_TRAILING and best_fav > 0 and (best_fav - pnl) >= TRAIL_GIVEUP:
                    actual_exit = d; stopped = True

                pos.loc[d] = sig * POSITION_SIZE
                if stopped:
                    break

            # Flat after exit until next signal
            start_idx = trading_index.get_loc(actual_exit)
            end_idx   = trading_index.get_loc(exit_plan)
            if isinstance(start_idx, slice): start_idx = start_idx.start
            if isinstance(end_idx, slice):   end_idx = end_idx.stop - 1
            if start_idx + 1 <= end_idx:
                after = trading_index[start_idx + 1: end_idx + 1]
                pos.loc[after] = 0.0

            # Record trade PnL
            trade_entries.append(entry)
            trade_exits.append(actual_exit)
            trade_pnls.append(sig * (px.loc[actual_exit] / entry_price - 1.0))

        # Daily returns
        strategy_daily = (pos * rets).fillna(0.0)

        # Costs on entries & exits
        commission_bp = 2
        per_side_cost = commission_bp / 10000.0
        cost_series = pd.Series(0.0, index=rets.index)
        for d in trade_entries:
            cost_series.loc[d] -= per_side_cost
        for d in trade_exits:
            cost_series.loc[d] -= per_side_cost

        strategy_daily_net = (strategy_daily + cost_series).astype(float)

        # Curves
        strategy_curve_net = (1 + strategy_daily_net).cumprod().rename(f"{ticker}_sentiment")
        bh_curve = (px / px.iloc[0]).rename(f"{ticker}_buyhold")

        # Store in results
        results[f"{ticker}_sentiment"] = strategy_curve_net
        results[f"{ticker}_buyhold"] = bh_curve

    return pd.DataFrame(results)