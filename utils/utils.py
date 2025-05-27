from typing import Any
import re
from urllib import parse
from sqlalchemy import create_engine, inspect, MetaData, inspect, Table, Column, select
import pyodbc
import pandas as pd
import shutil
import subprocess
import pickle
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
import psutil
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import matplotlib
from matplotlib import pyplot as plt
import time
import mlflow
from mlflow import log_metric, log_artifact, log_param, log_figure, log_metrics, log_params, start_run, end_run
from os import path, environ, listdir
import logging
from typing import Union, Any, Tuple
from itertools import product
import logging
from datetime import datetime
from random import randint, seed
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts, log_figure, MlflowClient, log_artifact, data
from os import path, listdir
import mlflow.data
import subprocess
import logging
import concurrent.futures
import psutil
import pandas as pd
import shutil
from os import environ

matplotlib.use('Agg')


def norm_zero_one(arr, expanding_window, negative=False):
    if negative == True:
        arr = -arr

    MAX = arr.expanding(expanding_window).max()

    if negative == True:
        NORM = - arr / MAX
    else:
        NORM = arr / MAX

    return NORM


def normalize_vector(arr):
    """
    Normalize a numpy array according to its L2 norm. 

    Parameters:
    -----------
    arr: np.ndarray or pd.Series
        The numpy array to be normalized.

    Returns:
    --------
    np.ndarray or pd.Series:
        The normalized numpy array.
    """
    norm = np.linalg.norm(arr)
    new = arr / norm
    return new

def grid(*iterables):
    """
    Generates a cartesian product grid from the provided iterables. Arguments can be a range, a tuple or a list or a set.
    If a tuple is provided, it is interpreted as arguments to the range function.
    If a range is provided, it is interpreted as a range object.
    If a list or a set is provided, it is interpreted as a list of values.

    Parameters
    ----------
    *iterables : range, tuple, or list
        Each argument can be a range, a tuple (interpreted as arguments to range), or a list of values.

    Returns
    -------
    list of tuple
        List containing tuples of all combinations from the cartesian product of the provided iterables.

    Examples
    --------
    >>> grid((1, 5, 2), [10, 20], range(2))
    [(1, 10, 0), (1, 10, 1), (1, 20, 0), (1, 20, 1), (3, 10, 0), (3, 10, 1), (3, 20, 0), (3, 20, 1)]
    """
    def helper(x):
        if isinstance(x, range):
            return list(x)
        elif isinstance(x, tuple):
            return list(range(*x))
        elif isinstance(x, list) or isinstance(x, set):
            return x
        else:
            raise TypeError(f"Unsupported type: {type(x)}. Expected range, tuple, or list.")
    
    all = list(map(helper, iterables))
    allgrid = list(product(*all))
    return allgrid

def rand_samples(df, n=10, iloc=True, sample_size=None, seed_=None):
    """
    Generates random samples of a DataFrame

    Parameters:
    ------------
    df : pd.DataFrame
        DataFrame to sample
    n : int
        Number of samples to generate
    iloc : bool
        If True, will return the iloc index of the samples. If False, will return the index of the samples
    sample_size : int
        Size of the samples. If None, will take 2/3 of the DataFrame size
    seed_ : int
        Seed for the random number generator
    Returns:
    --------
    samples : list
        List of tuples with the start and end of the samples
    >>> rand_samples(df, n=10)
    [(0, 19), (10, 39), (20, 49), (30, 59), (40, 69), (50, 79), (60, 89), (70, 99), (80, 109), (90, 119)]
    """
    seed(seed_)
    samples = []
    if sample_size is None:
        sample_size = int(len(df) * (2 / 3))
    else:
        sample_size = int(sample_size * len(df))

    while len(samples) < n:
        start = randint(0, len(df) - sample_size)
        end = start + sample_size
        samples.append((start, end))

    if iloc == True:
        return samples

    else:
        return list(map(lambda x: (df.iloc[x[0]: x[1]].index[0], df.iloc[x[0]: x[1]].index[-1]), samples))

def logret(ser: pd.Series, lag: int = 1):
    """
    Calculate log return of a series
    example:
    >>> logret(pd.Series([1, 2, 3, 4, 5]), lag=1)
    """
    RET = np.log(ser / ser.shift(lag))
    return RET


def SMA(ser: pd.Series, window: int = 1):
    ser = ser.rolling(window).mean()
    return ser


def col_ml(prefix, sufix, tag):
    if isinstance(sufix, tuple):
        sufix = sufix[0]
    elif isinstance(sufix, type(None)):
        sufix = None
    elif isinstance(sufix, str):
        sufix = sufix

    field = [prefix, sufix]
    if None in field:
        field.remove(None)

    sufpre = '_'.join(field)

    if tag is not None:
        col = (tag, sufpre)
    else:
        col = sufpre

    return col


def zscore(x: pd.Series, z_window):
    x_mean = x.rolling(z_window).mean()
    x_std = x.rolling(z_window).std()
    x_diff = x - x_mean
    z = x_diff / x_std
    return z


def p2z(p=0.5):
    return norm.ppf(p)


def zpos(Z: pd.Series, p_thres, name=None):
    z_thres = p2z(p_thres)
    POS_ACT = pd.Series(np.where(Z.isna(), np.nan, np.where(
        Z > z_thres, 1, np.where(Z < -z_thres, -1, 0))), index=Z.index)
    if name:
        POS_ACT.name = name
    else:
        POS_ACT.name = 'pos_act'
    return POS_ACT


def pos_scaled(ser: pd.Series, window=20):
    """
    Scales the positive and negative values of a pandas Series to a continous scale of [1,-1].

    Parameters:
    ----------
    ser: pd.Series
        The pandas Series to be scaled.
    window: int
        The window used to calculate the mean and standard deviation.
    """
    positive = norm_zero_one(ser.loc[ser > 0], window)
    negative = norm_zero_one(ser.loc[ser < 0], window, negative=True)
    zero = ser.loc[ser == 0]
    pos = pd.concat([positive, negative, zero])
    pos.sort_index(inplace=True)
    return pos

@dataclass
class Stats:
    """
    A dataclass that represents the statistics of a trading strategy.

    Attributes:
    -----------
    df : pd.DataFrame
        The DataFrame containing the strategy's returns.
    returns_total : float
        The total returns of the strategy.
    returns_mean : float
        The mean returns of the strategy.
    returns_total_annualized : float
        The annualized total returns of the strategy.
    returns_mean_annualized : float
        The annualized mean returns of the strategy.
    std : float
        The standard deviation of the strategy's returns.
    std_annualized : float
        The annualized standard deviation of the strategy's returns.
    sharpe_total : float
        The total Sharpe ratio of the strategy.
    sharpe_total_annualized : float
        The annualized total Sharpe ratio of the strategy.
    sharpe_mean_annualized : float
        The annualized mean Sharpe ratio of the strategy.
    drawdown : float, optional
        The maximum drawdown of the strategy.
    var90 : float, optional
        The 90% value at risk (VaR) of the strategy.
    """

    df: pd.DataFrame
    total_obs: int = None
    freq: str = None
    date_from: str = None
    date_to: str = None
    returns_total: float = None
    returns_mean: float = None
    returns_total_annualized: float = None
    returns_mean_annualized: float = None
    std: float = None
    std_annualized: float = None
    std_semi: float = None
    std_semi_annualized: float = None
    sortino: float = None
    sortino_annualized: float = None
    sharpe_total: float = None
    sharpe_total_annualized: float = None
    sharpe_mean_annualized: float = None
    calmar_mean_returns: float = None
    calmar_total_returns: float = None
    total_no_trade: int = None
    active_pos_ratio: float = None
    hits: float = None
    drawdown_days: float = None
    drawdown: float = None
    drawdown_annualized: float = None
    var90: float = None
    var95: float = None
    var99: float = None
    var90_param: float = None
    var95_param: float = None
    var99_param: float = None
    var90_annualized: float = None
    var95_annualized: float = None
    var99_annualized: float = None
    var90_param_annualized: float = None
    var95_param_annualized: float = None
    var99_param_annualized: float = None

    def print_stats(self, print_only=True):
        dict_stats = self.__dict__
        del dict_stats['df']

        for i in dict_stats.keys():
            if i != 'df' and 'annualized' in i or 'drawdown_days' in i or 'hits' in i:
                print(i, dict_stats[i])

        if print_only == False:
            return {k: v for k, v in dict_stats.items() if k != 'df' and 'annualized' in k or 'drawdown_days' in k or 'hits' in k}
        else:
            pass


@dataclass
class Factor:
    """
    A dataclass that represents the factor of a trading strategy.

    Attributes:
    ----------
    periods : int
        The number of periods used to calculate the factor.
    factor : float
        The value of the factor.
    frequency : str
        The frequency at which the factor is calculated.
    """
    periods: int
    factor: float
    frequency: str


class Metrics():
    """
    A class that represents the metrics of a trading strategy.

    Attributes:
    -----------
    df: pd.DataFrame
        A pandas DataFrame containing the price series. Index must be a datetime index.
    col_returns_log: str
        The name of the column containing the LOG! returns as it appears in the DataFrame. 
    strat_name: str
        The name of the series to be used. Will be used as a suffix to key columns of the DataFrame.

    Methods:
    --------
    log(cls, ser_price: pd.Series) -> pd.Series:
        Compute the log returns of a price series.

    cum(cls, ser_logret: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        Compute the cumulative log returns, simple returns and cumulative simple returns.

    cols_strat(self, df: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        Compute several columns used in strategy metrics calculations.

    drawdawn(cls, df, col_cumret: str) -> Tuple[pd.Series, pd.Series, pd.Series, Timestamp]:
        Compute the drawdowns of a cumulative returns series.

    factor_(cls, df) -> Factor:
        Compute the factor of the strategy.

    df_period(self, periods: int) -> pd.DataFrame:
        Return a DataFrame with the last n periods.

    stats(self, periods: int = None) -> Stats:
        Compute the statistics of the strategy.

    """

    def __init__(self, df, tag: str, logret: str = None, price: str = None, keep_cols: dict = None):

        if any((logret, price)) == False:
            raise ValueError(
                "At least one of the log returns or price series must be provided.")

        self._df = df.copy()
        self.tag = tag
        self._col_logret = logret
        self._keep_cols = keep_cols
        self._price = price

        self.col_logret, self.col_cumlogret, self.col_ret, self.col_cumret, self.col_cummax, self.col_drawdown, self.col_drawdown_days, self.col_drawdown_dates, _ = Metrics.col_names(
            tag=self.tag)

        if self._price is not None:
            self.col_price = (self.tag, 'price')

    @property
    def df(self):
        df = self._df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self._price is not None:
            df = df.dropna(subset=[self._price])
            LOGRET, CUM_LOGRET, RET, CUM_RET, CUM_MAX = Metrics.logret_all(
                df[self._price])
        else:
            df = df.dropna(subset=[self._col_logret])
            LOGRET = df[self._col_logret]
            CUM_LOGRET, RET, CUM_RET, CUM_MAX = Metrics.logret_cum(LOGRET)

        DD, DD_DATE, DD_DAYS = Metrics.drawdawn(df=None, col_cumret=CUM_RET)
        data = {
            self.col_logret: LOGRET,
            self.col_cumlogret: CUM_LOGRET,
            self.col_ret: RET,
            self.col_cumret: CUM_RET,
            self.col_cummax: CUM_MAX,
            self.col_drawdown: DD,
            self.col_drawdown_dates: DD_DATE,
            self.col_drawdown_days: DD_DAYS
        }
        df_new = pd.DataFrame(data)

        if self._price is not None:
            df_new.insert(loc=0, value=df[self._price], column=self.col_price)

        if self._keep_cols is not None:
            for col_name, attr_name in self._keep_cols.items():
                df_new[(self.tag, attr_name)] = df[col_name]
                df_new.dropna(inplace=True)
                setattr(self, f'col_{attr_name}', (self.tag, attr_name))

        return df_new.dropna()

    def stratinit(self, col_pos_realized: Union[str, pd.Series, np.ndarray] = None, strat_name: str = None, keep_cols: pd.DataFrame = None, decay: int = None):
        if strat_name is None:
            strategy_name = f'strat_{self.tag}'
        else:
            strategy_name = f'strat_{strat_name}_{self.tag}'

        Strategy = Metrics.Strategy()

        strat = Strategy(df=self.df, tag=strategy_name, logret=self.col_logret,
                         col_pos=col_pos_realized, keep_cols=keep_cols, logret_act=self.col_logret, decay=decay)
        setattr(self, strategy_name, strat)

        return strat

    @property
    def strat_main(self):
        strats = list(self.strats_all.keys())
        if len(strats) == 0:
            return None
        else:
            stratname = strats[0]
            stratmetric = self.__getattribute__(stratname)
            return stratmetric

    def df_stats(self, periods: int = None):
        stat = self.get_stats(periods=periods)

        df = pd.DataFrame(data=stat, index=[self.tag])
        return df

    def df_strats_all(self, periods: int = None):
        l = []
        for strat in self.strats_all:
            stratm = self.__getattribute__(strat)
            df = stratm.df_stats()
            l.append(df)

        df = pd.concat(l, axis=0)
        return df

    @classmethod
    def Strategy(cls):

        return MetricsStrat

    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)

    @property
    def strats_all(self):
        attr_names = [i for i in self.__dict__.keys()
                      if i.startswith('strat_')]
        attr_strats = {i: self.__getattribute__(i) for i in attr_names}
        return attr_strats

    @property
    def df_conso(self):
        df_main = self.df
        l_df = [self.strats_all[i].df for i in self.strats_all]
        l_df = [df_main] + l_df
        df_conso = pd.concat(l_df, axis=1)
        return df_conso

    def stats_df(self, periods):
        act = self.stats(periods).df
        strats = [self.strats_all[i].stats(
            periods).df for i in self.strats_all]
        l_df = [act] + strats
        df = pd.concat(l_df, axis=1)
        return df

    @classmethod
    def logret(cls, ser_price):
        """
        Compute the log returns of a price series.

        Parameters:
        -----------
        ser_price: pd.Series
            A pandas series of prices.

        Returns:
        --------
        LOGRET: pd.Series
            A pandas series of log returns.
        """
        LOGRET = np.log(ser_price / ser_price.shift(1))
        LOGRET.iloc[0] = 0
        return LOGRET

    @classmethod
    def logret_all(cls, ser_price):
        LOGRET = Metrics.logret(ser_price)
        CUM_LOGRET, RET, CUM_RET, CUM_MAX = Metrics.logret_cum(LOGRET)
        return LOGRET, CUM_LOGRET, RET, CUM_RET, CUM_MAX

    @classmethod
    def logret_cum(cls, ser_logret: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute the cumulative log returns, simple returns and cumulative simple returns.

        Parameters:
        -----------
        ser_logret: pd.Series
            A pandas series of log returns.

        Returns:
        --------
        CUM_LOGRET: pd.Series
            A pandas series of cumulative log returns.
        RET: pd.Series
            A pandas series of simple returns.
        CUM_RET: pd.Series
            A pandas series of cumulative simple returns.
        """
        CUM_LOGRET = ser_logret.cumsum()
        RET = np.exp(ser_logret) - 1
        CUM_RET = np.exp(CUM_LOGRET) - 1

        CUM_MAX = CUM_RET.expanding().max()

        return CUM_LOGRET, RET, CUM_RET, CUM_MAX

    @classmethod
    def semi_std(cls, returns: Union[np.ndarray, pd.Series]):
        """
        Compute the semi standard deviation of a series.

        Parameters:
        -----------
        returns: np.ndarray | pd.Series

        Returns:
        --------
        semi_std: float
            The semi standard deviation of the series.
        """
        downside = np.where(returns < 0, returns, 0)
        downside_deviation = np.sqrt(np.mean(downside**2))
        return downside_deviation

    @classmethod
    def sortino(cls, returns: Union[np.ndarray, pd.Series]):
        """
        Compute the Sortino ratio of a series.

        Parameters:
        -----------
        returns: np.ndarray | pd.Series
            The series of returns.
        periods: int
            The number of periods used to calculate the factor.

        Returns:
        --------
        sortino: float
            The Sortino ratio of the series.
        """
        semi_std = Metrics.semi_std(returns)
        return returns.mean() / semi_std

    @classmethod
    def col_names(cls, tag: str):

        col_logret = (tag, 'logret')

        col_cumlogret = (tag, 'cumlogret')

        col_ret = (tag, 'ret')

        col_cumret = (tag, 'cumret')

        col_cummax = (tag, 'cummax')

        col_drawdown = (tag, 'drawdown')

        col_drawdown_days = (tag, 'drawdown_days')

        col_drawdown_dates = (tag, 'drawdown_dates')

        col_pos_realized = (tag, 'pos_realized')

        return col_logret, col_cumlogret, col_ret, col_cumret, col_cummax, col_drawdown, col_drawdown_days, col_drawdown_dates, col_pos_realized

    @classmethod
    def drawdawn(cls, df: pd.DataFrame = None, col_cumret: Union[str, pd.Series] = None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Timestamp]:
        """
        Compute the drawdowns of a cumulative returns series.

        Parameters:
        -----------
        df: pd.DataFrame
            A pandas DataFrame containing the cumulative returns series. Index must be a datetime index.
        col_cumret: str
            The name of the column containing the cumulative returns series.

        Returns:
        --------
        cummax: pd.Series
            A pandas series of the cumulative maximum returns.
        drawdown_date: pd.Series
            A pandas series of the dates of the drawdowns.
        drawdown_days: pd.Series
            A pandas series of the durations of the drawdowns.
        max_drawdown: Timestamp
            The maximum drawdown
        """
        if isinstance(df, pd.DataFrame):
            CUMRET = df[col_cumret]
        else:
            CUMRET = col_cumret

        CUMMAX = CUMRET.expanding().max()

        drawdown = pd.Series(data=CUMMAX - CUMRET, index=CUMRET.index)
        drawdown_condition = CUMMAX == CUMRET
        drawdown_date = pd.Series(drawdown_condition.index.where(
            drawdown_condition, np.nan), index=CUMRET.index).fillna(method='ffill')

        if (drawdown_date == 0).all():
            drawdown_days = pd.Series([np.nan for i in range(len(df))])
        else:
            drawdown_days = CUMRET.index - drawdown_date

        return drawdown, drawdown_date, drawdown_days

    @classmethod
    def factor_(cls, df) -> Factor:
        """
        Compute the factor of the strategy.

        Parameters:
        -----------
        df: pd.DataFrame
            A pandas DataFrame containing the price series. Index must be a datetime index.

        Returns:
        --------
        factor: Factor
            A Factor object containing the factor of the strategy.
        """

        cond = list(filter(lambda x: isinstance(x, pd.Timestamp), df.index))

        time_diff_seconds = df.loc[cond].index.to_series(
        ).diff().mean().total_seconds()
        rounded_time_diff_seconds = abs(round(time_diff_seconds))
        rounded_time_diff = pd.to_timedelta(
            rounded_time_diff_seconds, unit='s')

        rounded_days = round(
            rounded_time_diff.total_seconds() / 60 / 60 / 24, 0)

        if pd.Timedelta('1D') <= rounded_time_diff < pd.Timedelta('7D') or rounded_days == 1:
            frequency = 'daily'
            periods = 252
            # Assuming 252 trading days in a year
            factor = periods / len(df.index)
        elif pd.Timedelta('7D') <= rounded_time_diff < pd.Timedelta('20D'):
            frequency = 'weekly'
            periods = 52
            factor = periods / len(df.index)  # Assuming 52 weeks in a year
        elif pd.Timedelta('20D') <= rounded_time_diff < pd.Timedelta('365D'):
            frequency = 'monthly'
            periods = 12
            factor = periods / len(df.index)  # Assuming 12 months in a year
        elif rounded_days == 0:
            frequency = f'intraday: {rounded_time_diff_seconds}-seconds'
            df['time'] = df.index
            df['date'] = df['time'].dt.date
            df_daily = df.groupby('date')['time'].count()
            periods = int(df_daily.mean()) * 252
            factor = periods / len(df.index)
        else:
            raise ValueError("Unable to infer frequency from the series.")

        factor = Factor(periods=periods, factor=factor, frequency=frequency)

        return factor

    @property
    def factor(self) -> Factor:
        """
        Compute the factor of the strategy.

        Returns:
        --------
        factor: Factor
            A Factor object containing the factor of the strategy.
        """
        return Metrics.factor_(self.df)

    def df_period(self, periods: int) -> pd.DataFrame:
        """
        Return a DataFrame with the last n periods.

        Parameters:
        -----------
        periods: int
            The number of periods to return. If periods > 0 then will return the first n periods, otherwise will return the last n periods

        Returns:
        --------
        df: pd.DataFrame
            A pandas DataFrame containing the last n periods.
        """
        if periods < 0:
            df_slice = self.df.iloc[periods:].copy()
        elif periods > 0:
            df_slice = self.df.iloc[:periods]

        if df_slice.isna().all().all():
            raise ValueError(
                f"The DataFrame is empty. Review the periods.\n {df_slice}")

        return df_slice

    @classmethod
    def position_stats(cls, df, col_pos):

        POS = df[col_pos].dropna()

        signs = Metrics.sign(POS)

        buy = signs.get(1, 0)
        sell = signs.get(-1, 0)
        no_pos = signs.get(0, 0)
        total_trades = POS.diff().abs().sum()
        total_obs = len(POS)

        stats_pos = dict(buy=buy, sell=sell, no_pos=no_pos,
                         total_obs=total_obs, total_trades=total_trades)
        return stats_pos

    @classmethod
    def total_no_trades(cls, df, col_pos):
        data = Metrics.position_stats(df, col_pos)['total_trades']
        return data

    @classmethod
    def active_position_ratio(cls, df, col_pos):
        data = Metrics.position_stats(df, col_pos)
        buy_and_sell = data['buy'] + data['sell']
        total_obs = data['total_obs']
        active_pos = buy_and_sell / total_obs
        return active_pos

    @classmethod
    def sign(csl, arr):
        signs = np.sign(arr).astype('int').value_counts().to_dict()
        return signs

    @classmethod
    def hits(cls, df, col_returns):

        signs = Metrics.sign(df[col_returns])
        # data = Metrics.position_stats(df, col_returns)
        hit = signs.get(1, 0)
        nohit = signs.get(-1, 0)
        nopos = signs.get(0, 0)

        total = np.sum([hit, nohit])

        if total == 0:
            return 0
        else:
            return hit / total

    def get_stats(self, periods: int = None) -> dict:
        stats = dict(
            filter(lambda x: x[0] != 'df', self.stats(periods).__dict__.items()))

        return stats

    def log_mlflow_stats(self, expid: int, metrics: dict = None, param: dict = None, artifact: Union[tuple, str] = None, prefix: dict = None, log_fig=False):
        """
        Logs the statistics of the strategy to MLflow.
        Params:
        -------
        expid: str
            The experiment ID to log the statistics to. Default is None.

        metrics: dict, optional
            A dictionary containing additional metrics to log. Default is None.

        param: dict, optional
            A dictionary containing additional parameters to log. Default is None.

        prefix: dict, optional
            A dictionary containing the prefix for the statistics. Default is None.

        artifact: tuple | str, optional
            A tuple containing the artifact URL and the artifact object to save. First element is the artifact URL, second element is the artifact object. Default is None.
            If str, the artifact URL is passed and the artifact object is the current object.
            If None, no artifact is logged. Default is None.
            >>> artifact = ('artifact.pkl', self)

        log_fig: bool, optional
            A boolean indicating whether to log the figure. Default is False.
        """
        logs, params = self.mlflow_stats(param=param, prefix=prefix)

        if isinstance(metrics, dict):
            logs.update(metrics)
        else:
            pass

        if expid:
            with start_run(experiment_id=expid):
                log_metrics(logs)
                # for k,v in logs.items():
                #     log_metric(f"{k}", v)

                log_params(params)
                # for k,v in params.items():
                #     log_param(f"{k}", v)

                if artifact:
                    if isinstance(artifact, tuple):
                        artifact_url = artifact[0]
                        artifact_obj = artifact[1]
                    else:
                        artifact_url = artifact
                        artifact_obj = self
                    save_pkl(artifact_url, artifact_obj)
                    log_artifact(artifact_url)
                if log_fig:
                    fig, ax = self.plot()
                    log_figure(fig, 'fig.png')
                    plt.close(fig)
            end_run()

        else:
            raise ValueError("Experiment ID is required.")

        return logs, params

    def plot(self, periods=None):
        df = self.stats(periods).df

        if hasattr(self, 'col_pos'):
            fig, axs = plt.subplots(
                nrows=2, ncols=1, figsize=(10, 20), sharex=True, dpi=100)
            df[self.col_cumret].plot(title='Cumulative Log Return', ax=axs[0])
            df[self.col_cummax].plot(figsize=(10, 5), ax=axs[0])

        else:
            fig, axs = plt.subplots(
                nrows=1, ncols=1, figsize=(10, 10), sharex=True, dpi=100)
            df[self.col_cumret].plot(title='Cumulative Log Return', ax=axs)
            df[self.col_cummax].plot(figsize=(10, 5), ax=axs)

        if hasattr(self, 'col_pos'):
            df[self.col_pos].plot(title='Trading Positions', ax=axs[1])
            axs[0].legend(['Cumulative Return', 'Cumulative Max Return'])
            axs[0].set_ylabel('Cumulative Return')
            axs[1].set_ylabel('Position')
            axs[1].set_xlabel('Date')

        else:
            axs.legend(['Cumulative Return', 'Cumulative Max Return'])
            axs.set_ylabel('Cumulative Return')

        fig.suptitle(f'{self.tag}', fontsize=10, y=1.02)

        fig.tight_layout()

        return fig, axs

    def mlflow_stats(self, periods=None, param=None, prefix=None):
        stats = self.get_stats(periods)
        if prefix:
            logs = {f'{prefix}_{k}': v for k, v in stats.items(
            ) if isinstance(v, int) or isinstance(v, float)}
            params = {f'{prefix}_{k}': v for k, v in stats.items(
            ) if not isinstance(v, int) and not isinstance(v, float)}
        else:
            logs = {k: v for k, v in stats.items() if isinstance(v, int)
                    or isinstance(v, float)}
            params = {k: v for k, v in stats.items() if not isinstance(
                v, int) and not isinstance(v, float)}

        periods = {'from': self.df.index[0].isoformat(
        ), 'to': self.df.index[-1].isoformat()}
        params.update({'periods': periods})

        if param:
            params.update(param)
        else:
            pass

        return logs, params

    def stats(self, periods: int = None) -> Stats:
        """
        Compute the statistics of the strategy.

        Parameters:
        -----------
        periods: int, optional
            The number of periods to consider. If None, all periods are considered.

        Returns:
        --------
        stats: Stats
            A Stats object containing the statistics of the strategy.
        """
        if periods is None:
            df = self.df.copy().dropna()
            logging.debug(
                f'Calculating statistics for {df.shape[0]:,} observations.')
            if df.shape[0] < 2:
                logging.error(
                    'Number of observations is less than 2: not enough observations to calculate statistics.')
                # raise Exception('Number of observations is less than 2: not enough observations to calculate statistics.')
            factor = self.factor
        else:
            df = self.df_period(periods).copy().dropna()
            logging.debug(
                f'Calculating statistics for {df.shape[0]:,} observations.')
            if df.shape[0] < 2:
                logging.error(
                    'Number of observations is less than 2: not enough observations to calculate statistics.')
                # raise Exception('Number of observations is less than 2: not enough observations to calculate statistics.')
            factor = Metrics.factor_(df)

        total_obs = len(df)

        if total_obs == 0:
            stats = Stats(df=df)
        else:
            freq = factor.frequency
            date_from = df.index[0].isoformat()
            date_to = df.index[-1].isoformat()
            returns_total = df[self.col_logret].sum()
            returns_mean = df[self.col_ret].mean()
            returns_total_annualized = (
                (1 + returns_total)**(factor.factor)) - 1
            returns_mean_annualized = returns_mean * factor.periods
            std = df[self.col_ret].std()
            std_semi = Metrics.semi_std(df[self.col_ret])
            std_annualized = df[self.col_ret].std() * np.sqrt(factor.periods)
            std_semi_annualized = std_semi * np.sqrt(factor.periods)
            max_drawdown_days = df[self.col_drawdown_days].max().days
            max_drawdown = df[self.col_drawdown].max()
            drawdown_annualized = max_drawdown / factor.periods
            sharpe_total = returns_total / std
            sharpe_total_annualized = returns_total_annualized / std_annualized
            sharpe_mean_annualized = returns_mean_annualized / std_annualized
            sortino = Metrics.sortino(df[self.col_ret])
            sortino_annualized = sortino * np.sqrt(factor.periods)
            calmar_total = returns_total_annualized / max_drawdown
            calmar_mean = returns_mean_annualized / max_drawdown
            hits = Metrics.hits(df, self.col_ret)
            try:
                total_no_trade = Metrics.total_no_trades(df, self.col_pos)
                active_pos_ratio = Metrics.active_position_ratio(
                    df, self.col_pos)
            except Exception as e:
                total_no_trade = None
                active_pos_ratio = None
            var90 = df[self.col_ret].quantile(0.1)
            var95 = df[self.col_ret].quantile(0.05)
            var99 = df[self.col_ret].quantile(0.01)

            var90_param = norm.ppf(0.1, returns_mean, std)
            var95_param = norm.ppf(0.05, returns_mean, std)
            var99_param = norm.ppf(0.01, returns_mean, std)

            var90_annualized = var90 * np.sqrt(factor.periods)
            var95_annualized = var95 * np.sqrt(factor.periods)
            var99_annualized = var99 * np.sqrt(factor.periods)

            var90_param_annualized = var90_param * np.sqrt(factor.periods)
            var95_param_annualized = var95_param * np.sqrt(factor.periods)
            var99_param_annualized = var99_param * np.sqrt(factor.periods)

            stats = Stats(
                df=df,
                total_obs=total_obs,
                freq=freq,
                date_from=date_from,
                date_to=date_to,
                returns_total=returns_total,
                returns_mean=returns_mean,
                returns_mean_annualized=returns_mean_annualized,
                returns_total_annualized=returns_total_annualized,
                std=std,
                std_semi=std_semi,
                std_annualized=std_annualized,
                std_semi_annualized=std_semi_annualized,
                sortino=sortino,
                sortino_annualized=sortino_annualized,
                sharpe_total=sharpe_total,
                sharpe_total_annualized=sharpe_total_annualized,
                sharpe_mean_annualized=sharpe_mean_annualized,
                calmar_mean_returns=calmar_mean,
                calmar_total_returns=calmar_total,
                hits=hits,
                total_no_trade=total_no_trade,
                active_pos_ratio=active_pos_ratio,
                drawdown=max_drawdown,
                drawdown_days=max_drawdown_days,
                drawdown_annualized=drawdown_annualized,
                var90=var90,
                var95=var95,
                var99=var99,
                var90_param=var90_param,
                var95_param=var95_param,
                var99_param=var99_param,
                var90_annualized=var90_annualized,
                var95_annualized=var95_annualized,
                var99_annualized=var99_annualized,
                var90_param_annualized=var90_param_annualized,
                var95_param_annualized=var95_param_annualized,
                var99_param_annualized=var99_param_annualized
            )

        return stats

    def __str__(self):
        return f"Metrics object for {self.tag}"

    def __repr__(self):
        return f"Metrics object for {self.tag}"

class MetricsStrat(Metrics):
    def __init__(self, df, tag: str, logret: str = None, price: str = None, keep_cols: pd.DataFrame = None, col_pos: str = None, logret_act: str = None, decay=None):
        super().__init__(df, tag, logret, price, keep_cols)
        self._col_logret_act = logret_act
        self._col_pos = col_pos
        self.col_logret, self.col_cumlogret, self.col_ret, self.col_cumret, self.col_cummax, self.col_drawdown, self.col_drawdown_days, self.col_drawdown_dates, self.col_pos = Metrics.col_names(
            tag=self.tag)
        self._df = df.copy()
        self._keep_cols = keep_cols
        self.decay = decay

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'decay'):
            self.decay = None  # Provide a default value if missing

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, '_keep_cols'):
            self._keep_cols = None  # Provide a default value if missing

    def decay_weights(self, arr):
        n = len(arr)
        alpha = 1/n * self.decay
        weights = np.array([1 - np.exp(-alpha * i) for i in range(n)])
        return weights

    @property
    def df(self):

        LOGRET = self._LOGRETSTRAT
        POS = self._POS
        CUM_LOGRET, RET, CUM_RET, CUM_MAX = Metrics.logret_cum(LOGRET)
        DD, DD_DATE, DD_DAYS = Metrics.drawdawn(col_cumret=CUM_RET)

        data = {
            self.col_logret: LOGRET,
            self.col_cumlogret: CUM_LOGRET,
            self.col_ret: RET,
            self.col_cumret: CUM_RET,
            self.col_cummax: CUM_MAX,
            self.col_drawdown: DD,
            self.col_drawdown_dates: DD_DATE,
            self.col_drawdown_days: DD_DAYS
        }
        df_new = pd.DataFrame(data)

        df_new = pd.merge(df_new, POS, left_index=True,
                          right_index=True, how='left', suffixes=('', '_x'))
        df_new.drop(
            columns=[i for i in df_new.columns if '_x' in i], inplace=True)
        df_new[self.col_pos].fillna(method='ffill', inplace=True)

        if self._keep_cols is not None:
            df_keep = self._keep_cols.copy()
            cols_old = self._keep_cols.columns.tolist()
            cols_new = [(self.tag, i) for i in cols_old]
            cols_midx = pd.MultiIndex.from_tuples(cols_new)
            df_keep.columns = cols_midx

            for i in df_keep.columns:
                df_new = df_new.merge(
                    df_keep[i], left_index=True, right_index=True, how='left', suffixes=('', '_x'))
                df_new.drop(
                    columns=[i for i in df_new.columns if '_x' in i[0]], inplace=True)

        return df_new.dropna()

    @property
    def _POS(self):
        if isinstance(self._col_pos, str):
            POS = self._df[self._col_pos]
        elif isinstance(self._col_pos, pd.Series):
            POS = self._col_pos
        POS.name = self.col_pos
        return POS

    @property
    def _LOGRETACT(self):
        LOGRET = self._df[self._col_logret_act]
        return LOGRET

    @property
    def _LOGRETSTRAT(self):
        LOGRET_STRAT = self._POS * self._LOGRETACT
        LOGRET_STRAT.name = self.col_logret
        if self.__dict__.get('decay', None) == None:
            return LOGRET_STRAT
        elif self.decay != None:
            weights = self.decay_weights(LOGRET_STRAT)
            LOGRET_STRAT_decay = LOGRET_STRAT * weights
            return LOGRET_STRAT_decay


def kill_mlflow_server():
    for process in psutil.process_iter():
        if 'mlflow' in process.name():
            print(f"killing {process.pid}: {process.name()}")
            process.kill()

def set_tracking_uri(backend_store, artifact, host='127.0.0.1', port=9999, ):
    mlflow.set_tracking_uri(backend_store)
    mlflow.set_tracking_uri(artifact)
    mlflow.set_tracking_uri(f"http://{host}:{port}")
    return mlflow.get_tracking_uri()

def set_mlflow_env(backend_store, artifact_location):
    if backend_store is not None:
        if 'MLFLOW_TRACKING_URI' in environ:
            del environ['MLFLOW_TRACKING_URI']
            environ['MLFLOW_TRACKING_URI'] = backend_store

        environ['MLFLOW_TRACKING_URI'] = backend_store
    if artifact_location is not None:
        if 'MLFLOW_ARTIFACT_ROOT' in environ:
            del environ['MLFLOW_ARTIFACT_ROOT']
            environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location
        environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location

def start_mlflow_server(port=9999, host='127.0.0.1', backend_store=None, artifact=None, start_server=False, kill_server=False):
    """"
    Example:
    --------
    >>> backend_store = r'file:///' + abspath(join(pardir, pardir, 'mlruns'))
    >>> print(backend_store)
    >>> file:///e:\khalilk\development\mlruns
    >>> start_mlflow_server(backend_store=backend_store)
    >>> # example using sqlite
    >>> backend_store = r'sqlite:///' + abspath(join(pardir, pardir, 'mlruns', 'mlruns.db'))
    >>> print(backend_store)
    >>> sqlite:///e:\khalilk\development\mlruns\mlruns.db
    >>> start_mlflow_server(backend_store=backend_store)
    """

    if kill_server:
        for process in psutil.process_iter():
            if 'mlflow' in process.name():
                print("MLflow server is already running. Restarting...")
                process.kill()
                break
    # if backend_store is not None:
    #     mlflow_server_command = f"mlflow server --host {host} --port {port} --backend-store-uri {backend_store}"
    # else:
    #     mlflow_server_command = f"mlflow server --host {host} --port {port}"

    args = {
    '--host': host,
    '--port': port,
    '--backend-store-uri': backend_store,
    '--default-artifact-root': artifact,
        }

    args_filter = dict(filter(lambda x: x[1] is not None, args.items()))
    mlflow_server_command = "mlflow server " + ' '.join([f'{k} {v}' for k, v in args_filter.items()])

    set_tracking_uri(backend_store=backend_store, artifact=artifact, host=host, port=port)

    if start_server == True:
        subprocess.Popen(mlflow_server_command, shell=True)
    else:
        print('please open a terminal and run the following command:')
        print(mlflow_server_command)
        input('Press any key to continue...')
        

    mlflow.config.enable_async_logging(enable=True)
    print(f"MLflow server started at {host}:{port}")
    print(f'URI: {mlflow.get_tracking_uri()}')
    return mlflow_server_command

def experiment_init(experiment_name:str, artifact_location:str=None)->str:
    """
    Create a new experiment if it does not exist. If it exists, return the experiment ID.

    Parameters:
    -----------
    experiment_name: str
        Name of the experiment

    artifact_location: str
        Location where the artifacts will be stored

    Returns:
    --------
    experiment_id: str
        The experiment
    """
    # mlflow_server_command = f"mlflow server --host {host} --port {port}"
    # mlflow.set_tracking_uri(f"http://localhost:{port}")
    # subprocess.Popen(mlflow_server_command, shell=True)
    
    try:
        if artifact_location is not None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id 
        print('created new experiment\n')
        print(f'Experiment {experiment_name} ID: {experiment_id}')
    except:
        mlflow.exceptions.RestException
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id 
        logging.info(f'Experiment {experiment_name} exists\n')
        logging.info(f'Experiment {experiment_name} ID: {experiment_id}')

    mlflow.config.enable_async_logging(enable=True)
    return experiment_id



class Mlflow_track:
    # CLIENT = mlflow.tracking.MlflowClient()
    
    FIG_HEIGHT = 500
    FIG_WIDTH = 1000

    def __init__(self, port=9999, host='127.0.0.1'):
        self.port = port
        self.host = host
        # self.backend_store = backend_store
        # self.artifact = artifact
        # self.start_server = start_mlflow_server(port=self.port, host=self.host, backend_store=self.backend_store, artifact=self.artifact, start_server=start_server)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=f"http://{self.host}:{self.port}")
        self.experiments = self.client.search_experiments()

    def get_run(self, run_id):
        return self.client.get_run(run_id)
    

    def search_runs(self, experiment_id):
        return self.client.search_runs(experiment_ids=[experiment_id])
    
    def cols_metrics(self, experiment_id):
        return list(self.client.search_runs(experiment_ids=1,max_results=1)[0].data.metrics.keys())
    
    def cols_params(self, experiment_id):
        return list(self.client.search_runs(experiment_ids=1,max_results=1)[0].data.params.keys())
    
    def runs(self, experiment_id):
        all_runs = []
        page_token = None

        while True:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                max_results=1000,  # Maximum allowed per request
                page_token=page_token  # Token for pagination
            )
            
            all_runs.extend(runs)
            
            # If there's no next page, break the loop
            if len(runs) < 1000:
                break
            
            # Set the page token for the next iteration
            page_token = runs.token  # Use runs.token for next page
        return all_runs
    
    def experiment_name(self, experiment_id):
        return self.client.get_experiment(experiment_id).name
    
    def df_runs(self, experiment_id):
        logging.info(f'Retrieving runs for experiment {self.experiment_name(experiment_id)}')
        runs = list(map(lambda x: x.to_dictionary(), self.runs(experiment_id)))
        runs = list(map(lambda x: flatten_dict(x), runs))
        df_runs = pd.DataFrame(runs)
        cols = df_runs.columns.tolist()
        cols = list(map(lambda x: x.split('.')[-1], cols))
        df_runs.columns = cols
        return df_runs
    
    @classmethod
    def rank_metric(cls, arr:pd.Series, reverse=False):
        if reverse==True:
            reverse = - arr
            return (reverse.rank(pct=True).round(2) * 100).dropna().sort_values(ascending=False)
        else:
            return (arr.rank(pct=True).round(2) * 100).dropna().sort_values(ascending=False)

    def get_ranked_df(self,cols):

        df = self.df_runs.copy()

        def rank_metric(df:pd.DataFrame, col:str, reverse=False):
            col_rank = f'RANK_{col}'
            if reverse==True:
                reverse = - df[col]
                df[col_rank] = (reverse.rank(pct=True).round(2) * 100)
            else:
                df[col_rank] = (df[col].rank(pct=True).round(2) * 100)
            return df, col_rank


        def rank_wa(df, cols):
            df['WA'] = df[cols].mean(axis=1)
            df = rank_metric(df, 'WA')[0]
            df.sort_values(by='RANK_WA', ascending=False, inplace=True)

            return df
        
        cols_rank = []
        for k  in cols.keys():
            c = k
            r = cols[k]
            df, col_rank = rank_metric(df, c, reverse=r)
            cols_rank.append(col_rank)

        df = rank_wa(df, cols_rank)



        return df
    
    def get_artifact_url(self, run_id):
        uri = self.CLIENT.get_run(run_id).info.artifact_uri
        url = uri.replace("mlflow-artifacts:", "mlartifacts")

        return url
    
    def get_artifact_pkl(self, run_id):
        artifact_url = self.get_artifact_url(run_id)

        # pickle files inside sub dir
        try:
            path_dir = list(filter(lambda x: x.is_dir, self.CLIENT.list_artifacts(run_id)))[0]._path
            path_pkl = path.relpath(path.join(artifact_url, path_dir))
            ls = listdir(path_pkl)
            ls_pkl = list(filter(lambda x: x.endswith('.pkl'), ls))

            pkl_dir_all = [read_pickle((path_pkl, i)) for i in ls_pkl]
        except Exception as e:
            pkl_dir_all = []

        # pickle files in root dir
        try:
            pkl_files = list(filter(lambda x:x.path.endswith('pkl'), self.CLIENT.list_artifacts(run_id)))
            pkl_file_names = list(map(lambda x: x.path, pkl_files))
            pkl_file_all = [read_pickle((artifact_url, i)) for i in pkl_file_names]
        except Exception as e:
            pkl_file_all = []

        pkl_all = pkl_dir_all + pkl_file_all
        if len(pkl_all) == 1:
            return pkl_all[0]
        else:
            return pkl_all
        
    @staticmethod
    def get_tracking_uri():
        return mlflow.get_tracking_uri()

def clear_deleted_exp(backenddb:str, artifact_location=None)->None:
    """
    Delete all experiments marked as deleted from the sqlite backend of mlflow
    """
    dbmlflow = SQLAlchemy_sqlite(backenddb)
    exp = dbmlflow.table('experiments')
    stmt = exp.select().where(exp.c.lifecycle_stage == 'deleted')
    deleted = dbmlflow.get_df(stmt).experiment_id.tolist()
    logging.info(f'{deleted} available for deletion')
    runs = dbmlflow.table('runs')
    deleted_runs = dbmlflow.get_df(runs.select()).run_uuid.unique().tolist()
    exps_dir = listdir(artifact_location)
    if len(exps_dir) == 0:
        logging.info(f'No directories found in {artifact_location}')
    else:
        logging.info(f'Deleting {deleted} directories from {artifact_location}')
        for i in exps_dir:
            logging.debug(f'Checking {i}')
            if i in deleted_runs:
                dir_path = path.join(artifact_location, i)
                shutil.rmtree(dir_path, ignore_errors=True)  # Removes non-empty directory
                logging.debug(f'Deleted {dir_path}')

    for i in dbmlflow.table_names():
        if 'experiment_id' in dbmlflow.get_col_names(i):
            logging.info(f'Deleting {deleted} experimets from {i}')
            table = dbmlflow.table(i)
            stmt = table.delete().where(table.c.experiment_id.in_(deleted))
            dbmlflow.connection.execute(stmt)
    dbmlflow.connection.commit()


def flatten_dict(d, parent_key=None, sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_pickle(dir: tuple):
    dir_root = path.relpath(path.join(*dir))
    dir_sub = path.relpath(path.join(path.pardir, *dir))
    try:
        with open(dir_root, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        with open(dir_sub, 'rb') as f:
            data = pickle.load(f)
    return data

def save_pkl(path_file, obj):
    with open(path_file, 'wb') as f:
        pickle.dump(obj=obj, file=f, protocol=5)
    logging.info(f'saved {path_file}')

def load_pkl(path_file):
    with open(path_file, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f'loaded {path_file}')
    return obj


class SQLAlchemy_loader:
    """
    Module to interact with a database using SQLAlchemy
    Parameters:
    ----------
    dbname: str
        The name of the database to connect to
    msql: bool (optional)
        If True, connects to a Microsoft SQL Server database
        False not implemented yet. should implement to connect to other types of databases

    Methods:
    --------
    table_names(schema_name: str = None)
        Returns a list of table names in the connected database
    view_names(schema_name: str = None)
        Returns a list of view names in the connected database
    table(table_name: str, schema_name: str = None)
        Returns a SQLAlchemy Table object
    exec(stmt, return_=False)
        Executes a sqlalchemy SQL statement and returns the result if return_ is True
    get_table_details(table: classmethod)
        Returns the details of a table
    get_df(sqlalchemy_stmt: classmethod, optimize=None)
        Returns a pandas DataFrame from a SQLAlchemy statement
        optimize: dict 
            A dictionary of columns to be optimized
            >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
    get_df_batch(sqlalchemy_stmt: classmethod, table: classmethod = None, batchsize: int = 100000, optimize=None)
        Returns a pandas DataFrame from a SQLAlchemy statement in batches
    get_col_details(table: classmethod)
        table: classmethod
            A SQLAlchemy Table object
        Returns the details of the columns in a table
        >>> tbl = db.table('table_name')
        >>> db.get_col_details(tbl)
    optimize_df(df: pd.DataFrame, optimize: dict)
        Optimizes a pandas DataFrame
        >>> df = db.get_df(tbl.select())
        >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
        >>> df_optimized = db.optimize_df(df, optimize)
    """

    """
    creates SQLAlchemy objects to be later used to interact with a database
    """

    def __init__(self, dbname: str, connection_string, msql=True):
        if msql == True:
            self.dbname = dbname
            self.connection_string = connection_string
            self.parsed = parse.quote_plus(self.connection_string)
            self.sqlalchemy_url = f'mssql+pyodbc:///?odbc_connect={self.parsed}'
            self.pyodbc_conn = pyodbc.connect(self.connection_string)
            self.engine = create_engine(self.sqlalchemy_url, fast_executemany=True, future=True)
            self.connection = self.engine.connect()
            self.inspector = inspect(self.engine)
            self.schemas = [i for i in self.inspector.get_schema_names()]
            self.metadata = MetaData()
        else:
            raise NotImplementedError

        print('\n##################################################')
        print(f'\nNow connected to database: {self.dbname} \n')
        print(
            f'Create an instance by connecting to one of the following Schemas: \n {self.schemas}')
        print('\n##################################################\n')

    def table_names(self, schema_name: str = None)->list:
        """
        Returns a list of table names in the connected database
        Parameters:
        ----------
        schema_name: str
            The name of the schema to connect to
        Returns:
        -------
        tbl_names: list
            A list of table names in the connected database
        Example:
        -------
        >>> db.table_names('Coffee')
        """
        tbl_names = [t for t in self.inspector.get_table_names(
            schema=schema_name)]
        print(tbl_names)
        return tbl_names

    def view_names(self, schema_name: str = None)->list:
        """
        Returns a list of view names in the connected database
        Parameters:
        ----------
        schema_name: str
            The name of the schema to connect to
        Returns:
        -------
        tbl_names: list
            A list of view names in the connected database
        Example:
        -------
        >>> db.view_names('Coffee')
        """
        tbl_names = [t for t in self.inspector.get_view_names(
            schema=schema_name)]
        print(tbl_names)
        return tbl_names
        # return tbl_names

    def table(self, table_name: str, schema_name: str = None):
        """
        Returns a SQLAlchemy Table object to be used in other methods specifically in sqlalchemy statements
        Parameters:
        ----------
        table_name: str
            The name of the table to connect to
        schema_name: str
            The name of the schema to connect to
        Returns:
        -------
        tbl: classmethod
            A SQLAlchemy Table object
        Example:
        -------
        >>> tbl = db.table('table_name', 'Coffee')
        """
        tbl = Table(table_name, self.metadata,
                    autoload_with=self.engine, schema=schema_name)
        print(f'connected to: -> {tbl}')
        return tbl

    def exec(self, stmt:str, commit=True, return_=False):
        """
        Executes a sqlalchemy statement as a string and returns the result if return_ is True
        Parameters:
        ----------
        stmt: str
            A SQL statement
        return_: bool
            If True, returns the result of the SQL statement
        Returns:
        -------
        result: classmethod
            The result of the SQL statement
        Example:
        -------
        >>> tbl = db.table('table_name', 'Coffee')
        >>> stmt = tbl.select()
        >>> cursor = db.exec(stmt)

        #### to be continued on how to operate the cursur ###
        """
        with self.connection as conn:
            logging.info(f'executing statement: {stmt}')
            if return_ == True:
                result = conn.execute(stmt)
                return result
            else:
                conn.execute(stmt)
                if commit == True:
                    conn.commit()
                else:
                    pass

    def get_table_details(self, table: classmethod)->dict:
        """
        Returns the details of a table
        Parameters:
        ----------
        table: classmethod
            A SQLAlchemy Table object
        Returns:
        -------
        details: dict
            A dictionary of the table details
        Example:
        -------
        >>> tbl = db.table('table_name', 'Coffee')
        >>> db.get_table_details(tbl)
        """
        print(repr(table))
        return table

    def get_df(self, sqlalchemy_stmt: classmethod, optimize=None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame from a SQLAlchemy statement
        Parameters:
        ----------
        sqlalchemy_stmt: classmethod
            A SQLAlchemy statement
        optimize: dict
            A dictionary of columns to be optimized
            >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
        Returns:
        -------
        df: pd.DataFrame
            A pandas DataFrame
        Example:
        -------
        >>> tbl = db.table('table_name', 'Coffee')
        >>> df = db.get_df(tbl.select())
        """

        cols = self.get_col_names(sqlalchemy_stmt)
        result = self.connection.execute(sqlalchemy_stmt).fetchall()
        if optimize != None:
            df = SQLAlchemy_loader.optimize_df(
                pd.DataFrame(result, columns=cols), optimize=optimize)
        else:
            df = pd.DataFrame(data=result, columns=cols)
        return df

    def get_df_batch(self, sqlalchemy_stmt: classmethod, table: classmethod = None, batchsize: int = 100000, optimize=None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame from a SQLAlchemy statement in batches
        Parameters:
        ----------
        sqlalchemy_stmt: classmethod
            A SQLAlchemy statement
        table: classmethod
            A SQLAlchemy Table object
        batchsize: int
            The size of the batch to fetch
        optimize: dict
            A dictionary of columns to be optimized
            >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
        Returns:
        -------
        df: pd.DataFrame
            A pandas DataFrame
        Example:
        -------
        >>> tbl = db.table('table_name', 'Coffee')
        >>> df = db.get_df_batch(tbl.select(), batchsize=100000, optimize=optimize)
        """
        result_proxy = self.connection.execution_options(
            stream_results=True).execute(sqlalchemy_stmt)
        batch_size = batchsize
        rows = []
        if table == None:
            sqlalchemy_tbl = sqlalchemy_stmt.get_final_froms()[0]
        else:
            sqlalchemy_tbl = table
        cols = self.get_col_names(sqlalchemy_tbl)

        counter = 0
        df = pd.DataFrame(columns=cols)
        while True:
            batch = result_proxy.fetchmany(batch_size)
            result = [row for row in batch]
            print(f'{len(batch):,} rows fetched')
            # rows.extend(result)
            if optimize != None:
                df_append = SQLAlchemy_loader.optimize_df(
                    pd.DataFrame(result, columns=cols), optimize=optimize)
            else:
                df_append = pd.DataFrame(result, columns=cols)
            print(
                f'appending {df_append.shape[0]:,} rows and {df_append.shape[1]} columns')
            df_new = pd.concat([df, df_append], ignore_index=True)
            df = df_new
            counter += len(batch)
            print(
                f'dataframe of size {df.shape[0]:,} rows and {df.shape[1]:,} columns')
            print('The CPU usage is: ', psutil.cpu_percent(4))
            print('RAM memory % used:', psutil.virtual_memory()[2])
            print('\n')

            if not batch:
                break
        # df = pd.DataFrame(rows, columns=cols)
        return df

    def get_col_details(self, table: classmethod) -> dict:
        if isinstance(table, str):
            table = self.table(table)
        else:
            table = table
        details = {c: c.type for c in table.columns}
        for i, j in details.items():
            print(i, j)
            return details

    def get_col_names(self, table: classmethod) -> list:
        if isinstance(table, str):
            table = self.table(table)
        else:
            table = table
        cols = [i.name for i in table.columns]
        return cols

    @staticmethod
    def optimize_df(df: pd.DataFrame, optimize: dict) -> pd.DataFrame:
        """
        Optimizes a pandas DataFrame
        Parameters:
        ----------
        df: pd.DataFrame
            A pandas DataFrame
        optimize: dict
            A dictionary of columns to be optimized
            >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
        Returns:
        -------
        df: pd.DataFrame
            A pandas DataFrame
        Example:
        -------
        >>> df = db.get_df(tbl.select())
        >>> optimize = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
        """
        logging.debug('Optimizing DataFrame')
        df = df
        for i, j in optimize.items():
            try:
                logging.debug(f'trying converting col {i} to type {j}')
                if j.startswith('float'):
                    df[i] = df[i].apply(lambda x: convert(x, float))
                    df[i] = df[i].astype(j)
                elif j.startswith('int'):
                    df[i] = df[i].apply(lambda x: convert(x, float))
                    df[i] = df[i].apply(lambda x: convert(x, int))
                    df[i] = df[i].astype(j)
                elif j.startswith('category'):
                    df[i] = df[i].apply(lambda x: convert(x, str))
                    df[i] = df[i].astype(j)
                elif j.startswith('date'):
                    df[i] = pd.to_datetime(df[i])
                else:
                    df[i] = df[i]
            except Exception as e:
                logging.error(f'failed converting col {i} to type {j}')
                print(e)
                pass
        return df

    def __repr__(self) -> str:
        d = self.connection_string
        pattern = r'Server=(.*?),\d+;'
        match = re.findall(pattern, d)[0]
        pattern = r'Server=tcp:(.*?),\d+;'
        details = dict(Server=match, Database=self.dbname)
        return f'Database Details: {details}'

class SQLAlchemy_sqlite(SQLAlchemy_loader):
    def __init__(self, db_dir:str):
        self.db_dir = db_dir
        self.engine = create_engine(f'sqlite:///{db_dir}')
        self.connection = self.engine.connect()
        self.inspector = inspect(self.engine)
        self.schemas = [i for i in self.inspector.get_schema_names()]
        self.metadata = MetaData()


def convert(some, func):
    try:
        new = func(some)
    except Exception as e:
        new = some
    return new




def kill_mlflow_server():
    for process in psutil.process_iter():
        if 'mlflow' in process.name():
            print(f"killing {process.pid}: {process.name()}")
            process.kill()

def set_tracking_uri(backend_store, artifact, host='127.0.0.1', port=9999, ):
    mlflow.set_tracking_uri(backend_store)
    mlflow.set_tracking_uri(artifact)
    mlflow.set_tracking_uri(f"http://{host}:{port}")
    return mlflow.get_tracking_uri()

def set_mlflow_env(backend_store, artifact_location):
    if backend_store is not None:
        if 'MLFLOW_TRACKING_URI' in environ:
            del environ['MLFLOW_TRACKING_URI']
            environ['MLFLOW_TRACKING_URI'] = backend_store

        environ['MLFLOW_TRACKING_URI'] = backend_store
    if artifact_location is not None:
        if 'MLFLOW_ARTIFACT_ROOT' in environ:
            del environ['MLFLOW_ARTIFACT_ROOT']
            environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location
        environ['MLFLOW_ARTIFACT_ROOT'] = artifact_location

def start_mlflow_server(port=9999, host='127.0.0.1', backend_store=None, artifact=None, start_server=False, kill_server=False):
    """"
    Example:
    --------
    >>> backend_store = r'file:///' + abspath(join(pardir, pardir, 'mlruns'))
    >>> print(backend_store)
    >>> file:///e:\khalilk\development\mlruns
    >>> start_mlflow_server(backend_store=backend_store)
    >>> # example using sqlite
    >>> backend_store = r'sqlite:///' + abspath(join(pardir, pardir, 'mlruns', 'mlruns.db'))
    >>> print(backend_store)
    >>> sqlite:///e:\khalilk\development\mlruns\mlruns.db
    >>> start_mlflow_server(backend_store=backend_store)
    """

    if kill_server:
        for process in psutil.process_iter():
            if 'mlflow' in process.name():
                print("MLflow server is already running. Restarting...")
                process.kill()
                break
    # if backend_store is not None:
    #     mlflow_server_command = f"mlflow server --host {host} --port {port} --backend-store-uri {backend_store}"
    # else:
    #     mlflow_server_command = f"mlflow server --host {host} --port {port}"

    args = {
    '--host': host,
    '--port': port,
    '--backend-store-uri': backend_store,
    '--default-artifact-root': artifact,
        }

    args_filter = dict(filter(lambda x: x[1] is not None, args.items()))
    mlflow_server_command = "mlflow server " + ' '.join([f'{k} {v}' for k, v in args_filter.items()])

    set_tracking_uri(backend_store=backend_store, artifact=artifact, host=host, port=port)

    if start_server == True:
        subprocess.Popen(mlflow_server_command, shell=True)
    else:
        print('please open a terminal and run the following command:')
        print(mlflow_server_command)
        input('Press any key to continue...')
        

    mlflow.config.enable_async_logging(enable=True)
    print(f"MLflow server started at {host}:{port}")
    print(f'URI: {mlflow.get_tracking_uri()}')
    return mlflow_server_command

def experiment_init(experiment_name:str, artifact_location:str=None)->str:
    """
    Create a new experiment if it does not exist. If it exists, return the experiment ID.

    Parameters:
    -----------
    experiment_name: str
        Name of the experiment

    artifact_location: str
        Location where the artifacts will be stored

    Returns:
    --------
    experiment_id: str
        The experiment
    """
    # mlflow_server_command = f"mlflow server --host {host} --port {port}"
    # mlflow.set_tracking_uri(f"http://localhost:{port}")
    # subprocess.Popen(mlflow_server_command, shell=True)
    
    try:
        if artifact_location is not None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        else:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id 
        print('created new experiment\n')
        print(f'Experiment {experiment_name} ID: {experiment_id}')
    except:
        mlflow.exceptions.RestException
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id 
        logging.info(f'Experiment {experiment_name} exists\n')
        logging.info(f'Experiment {experiment_name} ID: {experiment_id}')

    mlflow.config.enable_async_logging(enable=True)
    return experiment_id



class Mlflow_track:
    # CLIENT = mlflow.tracking.MlflowClient()
    
    FIG_HEIGHT = 500
    FIG_WIDTH = 1000

    def __init__(self, port=9999, host='127.0.0.1'):
        self.port = port
        self.host = host
        # self.backend_store = backend_store
        # self.artifact = artifact
        # self.start_server = start_mlflow_server(port=self.port, host=self.host, backend_store=self.backend_store, artifact=self.artifact, start_server=start_server)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=f"http://{self.host}:{self.port}")
        self.experiments = self.client.search_experiments()

    def get_run(self, run_id):
        return self.client.get_run(run_id)
    

    def search_runs(self, experiment_id):
        return self.client.search_runs(experiment_ids=[experiment_id])
    
    def cols_metrics(self, experiment_id):
        return list(self.client.search_runs(experiment_ids=1,max_results=1)[0].data.metrics.keys())
    
    def cols_params(self, experiment_id):
        return list(self.client.search_runs(experiment_ids=1,max_results=1)[0].data.params.keys())
    
    def runs(self, experiment_id):
        all_runs = []
        page_token = None

        while True:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                max_results=1000,  # Maximum allowed per request
                page_token=page_token  # Token for pagination
            )
            
            all_runs.extend(runs)
            
            # If there's no next page, break the loop
            if len(runs) < 1000:
                break
            
            # Set the page token for the next iteration
            page_token = runs.token  # Use runs.token for next page
        return all_runs
    
    def experiment_name(self, experiment_id):
        return self.client.get_experiment(experiment_id).name
    
    def df_runs(self, experiment_id):
        logging.info(f'Retrieving runs for experiment {self.experiment_name(experiment_id)}')
        runs = list(map(lambda x: x.to_dictionary(), self.runs(experiment_id)))
        runs = list(map(lambda x: flatten_dict(x), runs))
        df_runs = pd.DataFrame(runs)
        cols = df_runs.columns.tolist()
        cols = list(map(lambda x: x.split('.')[-1], cols))
        df_runs.columns = cols
        return df_runs
    
    @classmethod
    def rank_metric(cls, arr:pd.Series, reverse=False):
        if reverse==True:
            reverse = - arr
            return (reverse.rank(pct=True).round(2) * 100).dropna().sort_values(ascending=False)
        else:
            return (arr.rank(pct=True).round(2) * 100).dropna().sort_values(ascending=False)

    def get_ranked_df(self,cols):

        df = self.df_runs.copy()

        def rank_metric(df:pd.DataFrame, col:str, reverse=False):
            col_rank = f'RANK_{col}'
            if reverse==True:
                reverse = - df[col]
                df[col_rank] = (reverse.rank(pct=True).round(2) * 100)
            else:
                df[col_rank] = (df[col].rank(pct=True).round(2) * 100)
            return df, col_rank


        def rank_wa(df, cols):
            df['WA'] = df[cols].mean(axis=1)
            df = rank_metric(df, 'WA')[0]
            df.sort_values(by='RANK_WA', ascending=False, inplace=True)

            return df
        
        cols_rank = []
        for k  in cols.keys():
            c = k
            r = cols[k]
            df, col_rank = rank_metric(df, c, reverse=r)
            cols_rank.append(col_rank)

        df = rank_wa(df, cols_rank)



        return df
    
    def get_artifact_url(self, run_id):
        uri = self.CLIENT.get_run(run_id).info.artifact_uri
        url = uri.replace("mlflow-artifacts:", "mlartifacts")

        return url
    
    def get_artifact_pkl(self, run_id):
        artifact_url = self.get_artifact_url(run_id)

        # pickle files inside sub dir
        try:
            path_dir = list(filter(lambda x: x.is_dir, self.CLIENT.list_artifacts(run_id)))[0]._path
            path_pkl = path.relpath(path.join(artifact_url, path_dir))
            ls = listdir(path_pkl)
            ls_pkl = list(filter(lambda x: x.endswith('.pkl'), ls))

            pkl_dir_all = [read_pickle((path_pkl, i)) for i in ls_pkl]
        except Exception as e:
            pkl_dir_all = []

        # pickle files in root dir
        try:
            pkl_files = list(filter(lambda x:x.path.endswith('pkl'), self.CLIENT.list_artifacts(run_id)))
            pkl_file_names = list(map(lambda x: x.path, pkl_files))
            pkl_file_all = [read_pickle((artifact_url, i)) for i in pkl_file_names]
        except Exception as e:
            pkl_file_all = []

        pkl_all = pkl_dir_all + pkl_file_all
        if len(pkl_all) == 1:
            return pkl_all[0]
        else:
            return pkl_all
        
    @staticmethod
    def get_tracking_uri():
        return mlflow.get_tracking_uri()

def clear_deleted_exp(backenddb:str, artifact_location=None)->None:
    """
    Delete all experiments marked as deleted from the sqlite backend of mlflow
    """
    dbmlflow = SQLAlchemy_sqlite(backenddb)
    exp = dbmlflow.table('experiments')
    stmt = exp.select().where(exp.c.lifecycle_stage == 'deleted')
    deleted = dbmlflow.get_df(stmt).experiment_id.tolist()
    logging.info(f'{deleted} available for deletion')
    runs = dbmlflow.table('runs')
    deleted_runs = dbmlflow.get_df(runs.select()).run_uuid.unique().tolist()
    exps_dir = listdir(artifact_location)
    if len(exps_dir) == 0:
        logging.info(f'No directories found in {artifact_location}')
    else:
        logging.info(f'Deleting {deleted} directories from {artifact_location}')
        for i in exps_dir:
            logging.debug(f'Checking {i}')
            if i in deleted_runs:
                dir_path = path.join(artifact_location, i)
                shutil.rmtree(dir_path, ignore_errors=True)  # Removes non-empty directory
                logging.debug(f'Deleted {dir_path}')

    for i in dbmlflow.table_names():
        if 'experiment_id' in dbmlflow.get_col_names(i):
            logging.info(f'Deleting {deleted} experimets from {i}')
            table = dbmlflow.table(i)
            stmt = table.delete().where(table.c.experiment_id.in_(deleted))
            dbmlflow.connection.execute(stmt)
    dbmlflow.connection.commit()



class StrategyBase():

    def __init__(self, inst, df, col_backtest:str, col_predict:str, nickname:str=None, keep_cols:dict=None):
        logging.debug(f'Initializing {self.__class__.__name__} with {inst} - {col_backtest} - {col_predict}')
        self.inst = inst
        self._keepcols = keep_cols
        self._nickname = nickname
        self._col_predict = col_predict
        self._col_backtest = col_backtest
        self._df = df.copy()
        self.metrics = Metrics(df=self._df, price=self._col_backtest, keep_cols=self.keep_cols, tag=self.inst)
        self.col_backtest = self.metrics.col_logret
        self.df = self.metrics.df

    @property
    def col_predict(self):
        if self._col_backtest == self._col_predict:
            c = self.col_backtest
        else:
            c = (self.metrics.tag, self.flatten_tuple(self._col_predict))
        return c
    
    @property
    def nickname(self):
        if self._nickname is None:
            nick = f'{self.__class__.__name__.lower()}_predict{self.col_predict}'
        else:
            nick = f'{self._nickname.lower()}_predict{self.col_predict}'
        return nick
    
    @property
    def keep_cols(self):
        keepcols = {}
        if self._col_backtest == self._col_predict:
            for k,v in self.__dict__.items():
                if '_col' in k and '_col_backtest' not in k and '_col_predict' not in k:
                    if isinstance(v, list):
                        for i in v:
                            keepcols[i] = self.flatten_tuple(i)
                    else:
                        keepcols[v] = self.flatten_tuple(v)
        else:
            for k,v in self.__dict__.items():
                if '_col' in k and '_col_backtest' not in k:
                    if isinstance(v, list):
                        for i in v:
                            keepcols[i] = self.flatten_tuple(i)
                    else:
                        keepcols[v] = self.flatten_tuple(v)

        if self._keepcols is not None:
            keepcols.update(self._keepcols)
        else:
            pass
        return keepcols
    
    @staticmethod
    def flatten_tuple(val):
        if isinstance(val, tuple):
            l = list(val)
            l = list(map(str, l))
            flat = ".".join(l)
            return flat
        else:
            return val
        
    @staticmethod
    def pos(Y:pd.Series, name='pos_act', window:int=20, scaled=False):
        if scaled == True:
            POSITIVE = Y.loc[Y>0].copy()
            POSITIVE_POS = norm_zero_one(POSITIVE, expanding_window=window)

            NEGATIVE = Y.loc[Y<0].copy()
            NEGATIVE_POS = norm_zero_one(NEGATIVE, expanding_window=window, negative=True)
            
            ZERO = Y.loc[Y==0].copy()

            POS = pd.concat([POSITIVE_POS, NEGATIVE_POS, ZERO], axis=0)
            POS.sort_index(inplace=True)
        elif scaled == False:
            POS = np.sign(Y)

        else:
            raise ValueError('scaled must be True or False')
        POS.name = name
        POS.index = Y.index
        POS.sort_index(inplace=True)
        return POS
    
    def brute(self, range_z, range_p, scaled:list=[False]):
        Z = list(range_z)
        P = list(map(lambda x: x/100, list(range_p)))
        combo = list(product(Z, P, scaled))
        return combo
    
    @property
    def class_specs(self):
        specs = {
            'inst': self.inst,
            'col_backtest': self.col_backtest,
            'col_predict': self.col_predict,
        }
        return specs
    
    @staticmethod
    def repr_params(dict_):

        def flatten(text):
            if isinstance(text, tuple):
                text = list(map(str, text))
                text = "_".join(text)
                return text
            elif isinstance(text, list):
                text = list(map(str, text))
                text = "_".join(text)
                return text
            elif (isinstance(text, int) or isinstance(text, float)) and not isinstance(text, bool):
                text = str(int(round(text,0)))
                return text
            elif isinstance(text, bool):
                return str(text)
            else:
                return str(text)
        
        l_specs = [f'{k}{"".join(flatten(v))}'for k,v in dict_.items()]
        text = "_".join(l_specs)

        return text
    
    @staticmethod
    def params_text(params=locals()):
        params_filter = dict(filter(lambda x: x[0] != 'self', params.items()))
        params_repr = StrategyBase.repr_params(params_filter)
        return params_repr, params_filter

    @staticmethod
    def find_param(text, key):
        res = re.findall(rf'{key}([a-zA-Z0-9]+)', text)
        if len(res) == 0:
            return None
        else:
            return res[0]
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.class_specs})'
class MACD(StrategyBase):
    COLS_DEFAULT = {'pos_act': 'pos_act', 'ema1': 'ema1', 'ema2': 'ema2', 'macd': 'macd', 'signal': 'signal', 'histo': 'histo'}
    def __init__(self, inst:str, df:pd.DataFrame, col_backtest:str, col_predict:str, keep_cols:dict = None ,backtest_sample:float=0.8):
        self._backtest_sample = backtest_sample
        StrategyBase.__init__(self, inst=inst, df=df, col_backtest=col_backtest, col_predict=col_predict, keep_cols=keep_cols)
        self.total_obs = len(self._df)
        self.metrics_IN = Metrics(df=self.df_sample[0], price=self._col_backtest, tag=inst, keep_cols=self.keep_cols)
        self.metrics_OUT = Metrics(df=self.df_sample[1], price=self._col_backtest, tag=inst, keep_cols=self.keep_cols)
        self._df_grid_IN = None
        self._df_grid_OUT = None
        self._df_grid = None
        self._grid_backtest = None


    @property
    def _df_grid_conso(self):
        grid = [self._df_grid_IN, self._df_grid_OUT, self._df_grid]
        gridnona = list(filter(lambda x: isinstance(x, pd.DataFrame), grid))
        df_grid_conso = pd.concat(gridnona, axis=1, join='inner')
        df_grid_conso = df_grid_conso.iloc[:, ~df_grid_conso.columns.duplicated()]
        return df_grid_conso

    @property
    def backtest_sample(self):
        if self._backtest_sample is None:
            return float(0.8)
        elif self._backtest_sample:
            if self._backtest_sample > 0 and self._backtest_sample < 1:
                return float(self._backtest_sample)
            else:
                raise ValueError(f'Backtest sample must be between 0 and 1 - got {self._backtest_sample}')
        
    @property
    def df_sample(self):
        in_ = self.backtest_sample
        out_ = 1 - self.backtest_sample

        df_in = self._df.iloc[:int(self.total_obs*in_)]
        df_out = self._df.iloc[-int(self.total_obs*out_):]
        
        return df_in, df_out
    
    def run_strat(self, window_short=20, window_long=200, window_signal=20, sample:Union[str, None]='in', expid=None):

        window_short = int(window_short)
        window_long = int(window_long)
        window_signal = int(window_signal)
        
        if sample == 'in':
            metrics = self.metrics_IN
        elif sample == 'out':
            metrics = self.metrics_OUT
        elif sample is None:
            metrics = self.metrics
        else:
            raise ValueError(f'Sample must be in, out or None - got {sample}')
        
        df = metrics.df
        df_macd = df.copy().xs(key=self.inst, level=0, axis=1)
        cols_default = ['pos_act', 'ema1', 'ema2', 'macd',  'signal', 'histo']
        df_macd['ema1'] = df_macd[metrics.col_price[-1]].ewm(span=window_short).mean()
        df_macd['ema2'] = df_macd[metrics.col_price[-1]].ewm(span=window_long).mean()
        df_macd['macd'] = df_macd.ema1 - df_macd.ema2
        df_macd['signal'] = df_macd.macd.ewm(span=window_signal).mean()
        df_macd['histo'] = df_macd.macd - df_macd.signal
        df_macd['pos_act'] = self.pos(Y=df_macd['histo'], window=window_signal, scaled=True)
        df_macd['pos_realized'] = df_macd.pos_act.shift(1)
        stratmetrics = metrics.stratinit(df_macd['pos_realized'], strat_name=f'macd_{window_short}_{window_long}_{window_signal}', keep_cols=df_macd[cols_default])
        
        if expid:
            params = {'window_short': window_short, 'window_long': window_long, 'window_signal': window_signal}
            params_tolog = params.copy()
            params_tolog.update({'inst' : self.inst, 'col_predict': self.col_predict, 'col_backtest': self.col_backtest})
            stratmetrics.log_mlflow_stats(expid=expid, param=params_tolog, metrics=self.tstats, log_fig=True)
        
        return stratmetrics
    
    def brute(self, range_short, range_long, range_signal, sample, opt_stat='returns_total'):
        r = (range_short, range_long, range_signal)
        logging.info(f'backtesting ranges: {r} {sample} of sample')

        
        def back_test_wrapper(r):
            short, long, signal = r
            stratmetrics = self.run_strat(window_short=short, window_long=long, window_signal=signal, sample=sample)
            stats = stratmetrics.get_stats()
            sharpe = stats[opt_stat]
            return -sharpe
        
        res_opt = brute(func=back_test_wrapper, ranges=r, finish=None, full_output=True)

        params_short  = list(map(lambda x: int(x), res_opt[2][0].reshape(-1,1)))
        params_long = list(map(lambda x: int(x), res_opt[2][1].reshape(-1,1)))
        params_signal = list(map(lambda x: int(x), res_opt[2][2].reshape(-1,1)))
        res = res_opt[3].reshape(-1,1) * -1
        idx = pd.MultiIndex.from_tuples(tuple(zip(params_short,params_long, params_signal)), names=['short', 'long', 'signal'])
        cols = pd.MultiIndex.from_tuples([(sample, opt_stat)], names=['sample', 'opt_stat'])
        df_res = pd.DataFrame(data=res, index=idx, columns=[opt_stat])
        df_res.sort_values(by=opt_stat, ascending=False, inplace=True)
        df_res.dropna(inplace=True)
        df_res.columns = cols
        df_res[(sample, 'rank')] = Mlflow_track.rank_metric(df_res[(sample, opt_stat)])

        filter_longshort = df_res.index.get_level_values('short') < df_res.index.get_level_values('long')
        df_res = df_res.loc[filter_longshort]

        if sample == 'in':
            self._df_grid_IN = df_res
        elif sample == 'out':
            self._df_grid_OUT = df_res
        elif sample is None:
            self._df_grid = df_res
        return df_res
    
    def brute2(self, range_short, range_long, range_signal, opt_stat, sample=None, reverse=False, expid=None):
        if isinstance(opt_stat, str):
            opt_stat = [opt_stat]
        elif isinstance(opt_stat, list):
            logging.info(f'optimizing {opt_stat}')
        else:
            raise ValueError('opt_stat must be a string or a list of strings')

        for i in grid(range_short, range_long, range_signal):
            short, long, signal = i
            if short < long:
                self.run_strat(window_short=short, window_long=long, window_signal=signal, sample=sample, expid=expid)

        if sample is None:
            obj = 'metrics'
        else:
            obj = f'metrics_{sample.upper()}'

        df_stats = self.__getattribute__(obj).df_strats_all()
        params = pd.Series(df_stats.index).apply(lambda x: re.findall(r'\d+', x))
        params = np.array(list(map(lambda x: x[:3], params)))
        params = params.astype(int)
        param_name = ['short', 'long', 'signal']
        for i in range(params.shape[1]):
            df_stats.insert(i, param_name[i], params[:, i])


        df_stats.set_index(['short', 'long', 'signal'], inplace=True)
        df_stats = df_stats[opt_stat]
        if len(opt_stat) == 1:
            df_stats['rank'] = Mlflow_track.rank_metric(df_stats[opt_stat[0]])

        elif len(opt_stat) > 1: # if more than one opt_stat
            for n, i in enumerate(opt_stat):
                if isinstance(reverse, bool):
                    df_stats[f'rank_{i}'] = Mlflow_track.rank_metric(df_stats[i], reverse=reverse)
                elif isinstance(reverse, list): # if more than on sorting stat
                    if len(reverse) == len(opt_stat): # sorting stats mustt be equal to the number of opt_stat
                        df_stats[f'rank_{i}'] = Mlflow_track.rank_metric(df_stats[i], reverse=reverse[n])
                    else:
                        raise ValueError('Length of reverse must be equal to the number of opt_stat')

            df_stats['rank'] = df_stats[[f'rank_{i}' for i in opt_stat]].mean(axis=1)
            df_stats['rank'] = Mlflow_track.rank_metric(df_stats['rank'])
        cols = pd.MultiIndex.from_product([[sample], df_stats.columns], names=['sample', 'opt_stat'])
        df_stats.columns = cols
        # df_stats.reset_index(inplace=True)

        ## save attributes to class
        if sample == 'in':
            self._df_grid_IN = df_stats
        elif sample == 'out':
            self._df_grid_OUT = df_stats
        elif sample is None:
            self._df_grid = df_stats
    
    def backtest(self, range_short:str, range_long:str, range_signal:tuple, opt_stat:str, select_top=3):
        samples = ['in', 'out']
        for s in samples:
            self.brute(range_short=range_short, range_long=range_long, range_signal=range_signal, sample=s, opt_stat=opt_stat)

        df_grid = self._df_grid_conso.copy()

        df_grid[('variance', 'rank')] = np.abs(df_grid[('in', 'rank')] - df_grid[('out', 'rank')]) / np.abs(df_grid[('out', 'rank')])
        df_grid[('variance', 'rank')] = Mlflow_track.rank_metric(df_grid[('variance', 'rank')], reverse=True)
        df_grid[('wa', 'rank')] = Mlflow_track.rank_metric(df_grid[[('in', 'rank'), ('out', 'rank'), ('variance', 'rank')]].mean(axis=1))
        df_grid.sort_values(by=('wa', 'rank'), ascending=False, inplace=True)

        opt = df_grid.iloc[:select_top][('wa', 'rank')].index.tolist()
        for o in opt:
            self.run_strat(window_short=o[0], window_long=o[1], window_signal=o[2], sample=None)

        self._df_grid_backtest = df_grid

        return df_grid