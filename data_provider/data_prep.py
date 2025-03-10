import os
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf

##############################################
# DATA FETCHING AND FILTERING FROM YFINANCE
##############################################
def format_tickers(ticker_list):
    """Return a comma-separated string of tickers."""
    return ", ".join(ticker_list)

def fetch_and_save_ticker_data(ticker_list, output_csv, start_date, end_date, threshold=0.1):
    """
    Download stock data using yfinance, filter tickers based on zero-return ratio,
    remove consecutive duplicate days, and save the resulting DataFrame as a CSV.

    Parameters
    ----------
    ticker_list : list
        List of stock tickers.
    output_csv : str
        File path to save the filtered CSV.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    threshold : float
        Maximum allowed proportion of zero daily returns.
    """
    print(f"Fetching data for tickers (threshold={threshold})...")
    all_data = []
    for ticker in ticker_list:
        try:
            stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False)
            if stock_data.empty:
                print(f"No data for {ticker}, skipping...")
                continue
            stock_data.reset_index(inplace=True)
            # Use 'Adj Close' if available; otherwise, default to 'Close'
            if 'Adj Close' not in stock_data.columns:
                stock_data['Adj Close'] = stock_data['Close']
            # Select and rename relevant columns
            stock_data = stock_data[['Date', 'Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']].copy()
            stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'adjclose', 'volume']
            stock_data['tic'] = ticker
            # Normalize date (remove time component)
            stock_data['date'] = pd.to_datetime(stock_data['date'], utc=True).dt.normalize()
            all_data.append(stock_data)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if not all_data:
        print("No data fetched for any ticker.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df[['date', 'tic', 'open', 'close', 'high', 'low', 'adjclose', 'volume']]
    final_df.sort_values(by=['date', 'tic'], inplace=True)

    # Remove tickers with constant 'adjclose'
    constant_adjclose = final_df.groupby('tic')['adjclose'].nunique()
    abnormal_tickers = constant_adjclose[constant_adjclose == 1].index.tolist()
    if abnormal_tickers:
        print("Removing tickers with constant 'adjclose':", abnormal_tickers)
        final_df = final_df[~final_df['tic'].isin(abnormal_tickers)]

    # Remove tickers with a high proportion of zero daily returns
    final_df['daily_return'] = final_df.groupby('tic')['adjclose'].pct_change().fillna(0)
    final_df['is_zero_return'] = (final_df['daily_return'] == 0)
    zero_ratio_by_tic = final_df.groupby('tic')['is_zero_return'].mean()
    removed_tickers = zero_ratio_by_tic[zero_ratio_by_tic > threshold].index.tolist()
    print(f"Removing {len(removed_tickers)} tickers with zero-return ratio > {threshold*100:.1f}%.")
    final_df = final_df[~final_df['tic'].isin(removed_tickers)].copy()

    # Keep only dates where the maximum number of tickers is present
    ticker_counts = final_df.groupby('date')['tic'].nunique()
    expected_count = ticker_counts.max()
    complete_dates = ticker_counts[ticker_counts == expected_count].index
    final_df = final_df[final_df['date'].isin(complete_dates)].copy()

    # Remove consecutive duplicate days based on 'adjclose'
    pivot_df = final_df.pivot(index='date', columns='tic', values='adjclose')
    pivot_df.sort_index(inplace=True)
    consecutive_dup_mask = pivot_df.eq(pivot_df.shift(1)).all(axis=1)
    remaining_dates = pivot_df.index[~consecutive_dup_mask]
    final_df = final_df[final_df['date'].isin(remaining_dates)]
    final_df.drop(['daily_return', 'is_zero_return'], axis=1, inplace=True)

    print(f"Saving filtered raw data to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("Raw data saved.")


##############################################
# FEATURE GENERATION USING YFINANCE DATA
##############################################
class YfinancePreprocessor:
    def __init__(self, input_path, output_path):
        """
        Initialize the preprocessor with input and output file paths.
        """
        self.input_path = input_path
        self.output_path = output_path

    def make_feature(self):
        """
        Read raw CSV from input_path, generate additional features,
        and save the processed DataFrame to output_path.
        Creates 'z*' columns (e.g., zopen, zhigh) and 'zd_*' columns for rolling means.
        """
        try:
            df = pd.read_csv(self.input_path)
        except Exception as e:
            print(f"Error reading input file {self.input_path}: {e}")
            return

        eps = 1e-12  # Prevent division by zero
        periods = [5, 10, 15, 20, 25, 30, 60]

        df["zopen"] = df["open"] / (df["close"] + eps) - 1
        df["zhigh"] = df["high"] / (df["close"] + eps) - 1
        df["zlow"] = df["low"] / (df["close"] + eps) - 1
        df["zadjcp"] = df["adjclose"] / (df["close"] + eps) - 1
        df["zclose"] = df.groupby('tic')['close'].pct_change()

        for period in periods:
            rolling_mean = df.groupby('tic')['close'].transform(lambda x: x.rolling(window=period).mean())
            df[f'zd_{period}'] = rolling_mean / df['close'] - 1

        try:
            df.to_csv(self.output_path, index=False)
            print(f"Processed data saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

        return df

