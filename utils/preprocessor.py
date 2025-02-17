import pandas as pd
import numpy as np
import abc
from typing import Union, Text, Optional
from scipy.stats import zscore as standard_zscore

EPS = 1e-12


def moe_label(df, pred_lens):
    """
    Generate MOE labels by calculating cumulative returns for multiple prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with index ['date', 'tic'] and a 'close' column.
    pred_lens : list of int
        List of prediction intervals (e.g., [1, 5, 20]).

    Returns
    -------
    moe_label : pd.Series
        A Series of MOE labels as integers (e.g., [0, 1, 2]) corresponding to the best prediction interval for each date.
    """
    df = df.sort_index()

    # Determine the maximum prediction interval and calculate the number of liquidation periods
    max_len = max(pred_lens)
    cumulative_returns = {}

    for pred_len in pred_lens:
        num_intervals = max_len // pred_len  # Number of liquidation periods within the maximum period
        portfolio_value = pd.Series(1, index=df.index)

        for i in range(num_intervals):
            # Get start and end prices for each interval
            shift_start = df.groupby("tic")["close"].shift(-i * pred_len)
            shift_end = df.groupby("tic")["close"].shift(- (i + 1) * pred_len)
            interval_return = (shift_end - shift_start) / shift_start
            interval_return = interval_return.fillna(0)

            # Average the returns across stocks for each date and update portfolio value
            daily_gains = (1 + interval_return).groupby(level='date').mean()
            portfolio_value.loc[daily_gains.index] *= daily_gains

        cumulative_returns[pred_len] = portfolio_value.groupby(level='date').sum()

    # Combine cumulative returns into a DataFrame
    cumulative_returns_df = pd.DataFrame(cumulative_returns)

    # Select the prediction interval with the highest cumulative return as the MOE label
    moe_label_series = cumulative_returns_df.idxmax(axis=1)
    moe_label_series = moe_label_series.map({val: idx for idx, val in enumerate(pred_lens)})
    moe_label_series = moe_label_series.dropna().astype(int)
    return moe_label_series


def generate_labels_single(df, lookahead=5):
    """
    Generate single-horizon return labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'close' column.
    lookahead : int
        Number of periods ahead to compute the return.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with an additional 'return_ratio' column.
    """
    df = df.sort_index()
    df["close_current"] = df.groupby("tic")["close"].shift(0)
    df["close_future"] = df.groupby("tic")["close"].shift(-lookahead)
    df["return_ratio"] = (df["close_future"] - df["close_current"]) / df["close_current"]
    df = df.drop(columns=["close_current", "close_future"])
    return df.dropna(subset=["return_ratio"])


def generate_labels_multiple_lookaheads(df, lookaheads=[1, 5, 20]):
    """
    Generate multiple return labels for different horizons.

    For each lookahead horizon in lookaheads, a new column 'return_ratio_<lh>' is created.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'close' column.
    lookaheads : list of int
        List of lookahead intervals (e.g., [1, 5, 20]).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with additional return ratio columns.
    """
    df = df.sort_index()
    df["close_current"] = df.groupby("tic")["close"].shift(0)
    for lh in lookaheads:
        df_future = df.groupby("tic")["close"].shift(-lh)
        df[f"return_ratio_{lh}"] = (df_future - df["close_current"]) / df["close_current"]
    df = df.drop(columns=["close_current"])
    label_cols = [f"return_ratio_{lh}" for lh in lookaheads]
    return df.dropna(subset=label_cols)


def get_level_index(df: pd.DataFrame, level: Union[str, int]) -> int:
    """
    Helper function to get the numeric index of a level in a MultiIndex.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check.
    level : Union[int, str]
        The level name (str) or index (int).

    Returns
    -------
    int
        The numeric index of the level.
    """
    if isinstance(level, int):
        return level
    elif isinstance(level, str):
        if level in df.index.names:
            return df.index.names.index(level)
        raise ValueError(f"Level '{level}' not found in MultiIndex names: {df.index.names}")
    else:
        raise TypeError(f"Level must be an integer or string, got {type(level)}")


def fetch_df_by_index(
        df: pd.DataFrame,
        selector: Union[pd.Timestamp, slice, str, list, pd.Index],
        level: Union[str, int],
        fetch_orig: bool = True,
) -> pd.DataFrame:
    """
    Fetch data from a DataFrame using the given selector and level.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to fetch data from.
    selector : Union[pd.Timestamp, slice, str, list, pd.Index]
        The index selector for filtering.
    level : Union[int, str]
        The level in the MultiIndex to filter on.
    fetch_orig : bool, optional
        If True, return the original DataFrame if no filtering is performed.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    if level is None or isinstance(selector, pd.MultiIndex):
        return df.loc[selector]

    level_idx = get_level_index(df, level)

    if level_idx == 0:
        idx_slc = (selector, slice(None, None))
    else:
        idx_slc = (slice(None, None), selector)

    if fetch_orig:
        for slc in idx_slc:
            if slc != slice(None, None):
                return df.loc[pd.IndexSlice[idx_slc]]
        return df
    else:
        return df.loc[pd.IndexSlice[idx_slc]]


def get_group_columns(df: pd.DataFrame, group: Union[Text, None]):
    """
    Get a group of columns from a DataFrame with MultiIndex columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multi-level columns.
    group : str
        The name of the feature group (i.e., the first level value).

    Returns
    -------
    Index
        Columns corresponding to the group.
    """
    if group is None:
        return df.columns
    else:
        return df.columns[df.columns.get_loc(group)]


class RobustZScoreNorm:
    """
    Robust Z-Score Normalization using the median and MAD.

    This method uses:
        - mean = median(x)
        - std = MAD(x) * 1.4826

    Parameters
    ----------
    fit_start_time : str
        The start time for fitting.
    fit_end_time : str
        The end time for fitting.
    fields_group : list[str], optional
        List of column names to normalize (default is all columns).
    clip_outlier : bool, optional
        Whether to clip outliers to [-3, 3] (default is True).
    """

    def __init__(self, fit_start_time, fit_end_time, fields_group=None, clip_outlier=True):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.mean_train = None
        self.std_train = None

    def fit(self, df: pd.DataFrame):
        if self.fit_start_time >= self.fit_end_time:
            raise ValueError("fit_start_time must be earlier than fit_end_time.")

        df_fit = df.loc[self.fit_start_time:self.fit_end_time]
        self.cols = self.fields_group if self.fields_group is not None else df_fit.columns.tolist()
        if not set(self.cols).issubset(df_fit.columns):
            raise ValueError(f"fields_group contains invalid columns: {self.fields_group}")

        X = df_fit[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train = mad * 1.4826
        self.std_train = np.maximum(self.std_train, 1e-8)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_train is None or self.std_train is None:
            raise ValueError("Call the fit method before using this transformer.")

        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train

        if self.clip_outlier:
            X = np.clip(X, -3, 3)

        df[self.cols] = X
        return df


class CSZScoreNorm:
    """
    Cross-Sectional Z-Score Normalization with optional outlier removal.

    Normalizes data for each time group (e.g., each date) using either standard or robust Z-Score.
    Optionally removes the top and bottom 5% of values before normalization.

    Parameters
    ----------
    fields_group : list[str]
        List of column names to normalize.
    method : str, optional
        Normalization method ("zscore" by default or "robust").
    remove_outliers_flag : bool, optional
        Whether to remove the top and bottom 5% of values (default is False).
    """

    def __init__(self, fields_group=None, method="zscore", remove_outliers_flag=False):
        if fields_group is None or not isinstance(fields_group, list):
            raise ValueError("fields_group must be a non-empty list of column names.")
        self.fields_group = fields_group
        self.remove_outliers_flag = remove_outliers_flag
        self.zscore_func = standard_zscore

    def remove_outliers_from_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Remove the top and bottom 5% of values for a column."""
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("The DataFrame must have a MultiIndex with 'date' and 'tic'.")
        if 'date' not in df.index.names or 'tic' not in df.index.names:
            raise ValueError("The MultiIndex must contain 'date' and 'tic' levels.")

        df_norm = df.copy()
        for col in self.fields_group:
            if col not in df_norm.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

            if self.remove_outliers_flag:
                df_norm = self.remove_outliers_from_column(df_norm, col)

            df_norm[col] = (
                df_norm.groupby(level="date", group_keys=False)[col]
                .transform(self.zscore_func)
            )
        return df_norm


class CSRankNorm:
    """
    Cross-Sectional Rank Normalization for MultiIndex DataFrames.

    Normalizes data for each time group by ranking values within each group,
    converting ranks to a uniform distribution, and scaling to approximate a standard normal distribution.

    Parameters
    ----------
    fields_group : Union[str, list[str]]
        Column name or list of column names to normalize.

    Example
    -------
    >>> processor = CSRankNorm(fields_group="feature1")
    >>> normalized_df = processor(df)
    """

    def __init__(self, fields_group):
        if isinstance(fields_group, str):
            fields_group = [fields_group]
        if not isinstance(fields_group, list) or len(fields_group) == 0:
            raise ValueError("fields_group must be a non-empty list or a single column name as a string.")
        self.fields_group = fields_group

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("The DataFrame must have a MultiIndex with 'date' and 'tic'.")
        if 'date' not in df.index.names or 'tic' not in df.index.names:
            raise ValueError("The MultiIndex must contain 'date' and 'tic' levels.")

        df_norm = df.copy()
        for col in self.fields_group:
            if col not in df_norm.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            ranks = df_norm.groupby(level="date", group_keys=False)[col].rank(pct=True)
            ranks -= 0.5  # Center around 0
            ranks *= 3.46  # Scale to approximate unit standard deviation
            df_norm[col] = ranks
        return df_norm
