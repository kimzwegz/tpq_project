import re
from scipy.stats import t
from os import listdir, path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import rand_samples, MACD, grid
from scipy.stats import ttest_ind


class BackTest:
    """
    A comprehensive backtesting and statistical analysis framework for evaluating MACD-based trading strategies.

    This class provides methods to:
        - Load, preprocess, and visualize historical market data for multiple instruments
        - Run brute-force grid search backtests on MACD parameter combinations
        - Split data into in-sample (80%) and out-of-sample (20%) sets for validation
        - Perform randomized sub-sampling within the in-sample period to assess parameter robustness
        - Summarize and compare performance metrics (e.g., Sharpe ratio, returns) across evaluation methods
        - Conduct statistical tests to evaluate overfitting and generalization stability
        - Generate comparative visualizations of in/out-of-sample and randomized test results

    The design supports:
        * Parameter tuning via exhaustive grid search over MACD short, long, and signal EMA windows
        * Aggregated pooled Sharpe calculations across randomized in-sample folds
        * Detailed performance summaries per instrument and cross-instrument comparisons
        * Statistical validation using Welch's t-tests

    Attributes
    ----------
    folder_main : str
        Path to the main project data directory.
    folder_data : str
        Subdirectory containing raw instrument data (parquet files).
    folder_backtest : str
        Subdirectory containing backtest result files.
    folder_backtest_in : str
        Subdirectory for storing in-sample backtest results.
    folder_backtest_out : str
        Subdirectory for storing out-of-sample backtest results.
    folder_backtest_random : str
        Subdirectory for storing randomized in-sample backtest results.
    insts_all : list of str
        List of instrument identifiers available in the dataset.

    Methods
    -------
    path_data(inst)
        Returns the file path to raw data for a given instrument.
    path_backtest(inst)
        Returns the file path to backtest results for a given instrument.
    df_inst_raw(inst)
        Loads raw data for a single instrument.
    df_raw
        Loads and concatenates raw data for all instruments.
    df_yearinst
        Returns a year-by-instrument count of observations.
    df_inst
        Returns a count of observations per instrument.
    df_year
        Returns a year-by-year count of observations.
    df_pivot
        Returns a pivoted DataFrame with instruments as columns and years as index.
    plot_data()
        Plots data availability across instruments and years.
    descstats_details
        Returns descriptive statistics (count, mean, min, max) for data availability.
    descstats_summary
        Returns a summary table of data counts and yearly observations.
    run_backtest(...)
        Runs a brute-force MACD backtest for a given instrument or dataset.
    run_random_backtest(...)
        Runs multiple randomized in-sample MACD backtests for robustness checks.
    run_backtest_all(...)
        Executes the full backtesting process for all instruments.
    get_stats(inst)
        Compares in-sample and out-of-sample backtest metrics for an instrument.
    get_stats_rand(inst)
        Aggregates and ranks pooled Sharpe metrics from randomized in-sample runs.
    summarize_stats(inst, top_param)
        Summarizes performance of a top parameter set for an instrument.
    df_top_inout(...)
        Evaluates top-performing parameters for in/out-sample or randomized tests.
    compare(...)
        Compares rank and return differences between simple and randomized sampling.
    plot_compare(...)
        Plots comparative charts for rank and return differences.
    ttest(...)
        Performs Welch's t-tests on rank and return differences to assess significance.
    """
    def __init__(self):
        # File paths and constants
        self.folder_main = r'./data_cqf'
        self.folder_data = 'data' # data folder already loaded with all instruments; each instrument is a separate file
        self.folder_backtest = 'backtest' # backtest folder with all backtest results
        self.folder_backtest_in = r'backtest/in' # backtest of grid search results for the 80% of the original insample data
        self.folder_backtest_out = r'backtest/out' # backtest of grid search results for the 20% of the original outsample data
        self.folder_backtest_random = r'../data_cqf/backtest_random' # backtest of random samples performed over the 80% of the original insample data
        self.insts_all = self._insts_all()  # Load all instruments from the data folder

    def path_data(self, inst):
        filename = f'{inst}.parquet'
        path_data = path.join(self.folder_main, self.folder_data, filename)
        return path_data
    
    def path_backtest(self, inst):
        path_backtest = path.join(self.folder_main, self.folder_backtest, f'{inst}.h5')
        return path_backtest
    
    def df_inst_raw(self, inst):
        # Load instrument data from parquet file
        path_data = self.path_data(inst)
        df = pd.read_parquet(path_data)
        return df

    @property
    def df_raw(self):
        # Load raw data from parquet files
        df_raw = pd.concat(list(map(lambda x: pd.read_parquet(path.join(self.folder_main, self.folder_data, f'{x}.parquet')), self.insts_all))) ## reading parquet files for all and concatenating them into a single DataFrame
        df_raw['year'] = df_raw.index.year
        return df_raw
    
    @property
    def df_yearinst(self):
        # Group by year and instrument, counting the number of mid observations
        df_yearinst = self.df_raw.groupby(['year', 'inst'])['mid'].count().reset_index()
        return df_yearinst
    
    @property
    def df_inst(self):
        # Group by instrument, counting the number of mid observations
        df_inst = self.df_raw.groupby(['inst'])['mid'].count().reset_index()
        df_inst.rename(columns={'mid': 'count'}, inplace=True)
        return df_inst
    
    @property
    def df_year(self):
        # Group by year, counting the number of mid observations
        df_year = self.df_raw.groupby(['year'])['mid'].count().reset_index().set_index('year')
        df_year.rename(columns={'mid': 'count'}, inplace=True)
        return df_year
    
    @property
    def df_pivot(self):
        # Pivot the data so each instrument is a column, index is year, values are mid counts
        df_pivot = self.df_yearinst.pivot(index='year', columns='inst', values='mid')
        return df_pivot
    
    def _insts_all(self):
        # Load all instruments from the data folder
        insts_all = list(map(lambda x: x.split('.')[0], listdir(path.join(self.folder_main, self.folder_data)))) ## all instruments
        print(f'Instruments loaded: {len(insts_all)}\n {insts_all}')
        return insts_all
    
    def plot_data(self):
    # Loading data and preparing DataFrames

        # Pivot the data so each instrument is a column, index is year, values are mid counts
        df_pivot = self.df_pivot
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
        # Plot stacked bar chart
        df_pivot.plot(kind='bar', stacked=True, ax=ax[0, 0])
        self.df_inst.plot.bar(x='inst', y='count', title='Instrument Data Availability', ax=ax[0, 1], legend=False)
        self.df_yearinst.pivot(index='year', columns='inst', values='mid').plot.area(stacked=True, ax=ax[1, 0], title='Yearly Instrument Data Availability')
        self.df_year.plot(title='Yearly Data Availability', ax=ax[1, 1])
        fig.suptitle('Instrument Data Availability', fontsize=20)

        return df_pivot
    
    @property
    def descstats_details(self):
        # Describe the pivoted DataFrame
        df_pivot_describe = self.df_pivot.describe().loc[['count', 'mean', 'min', 'max']]
        df_pivot_describe.index=['no yrs', 'average observations/year', 'min observations/year', 'max observations/year']
        return df_pivot_describe
    
    @property
    def descstats_summary(self):
        # Describe the pivoted DataFrame
        summary = [
            {'instruments': len(self.insts_all), 'observations': len(self.df_raw), 'observations per year': self.df_raw.groupby(['year'])['mid'].count().mean()},
            {'instruments': "_", 'observations': self.df_raw.groupby(['year'])['mid'].count().min(), 'observations per year': self.df_raw.groupby(['year', 'inst'])['mid'].count().min()},
            {'instruments': "_", 'observations': self.df_raw.groupby(['year'])['mid'].count().max(), 'observations per year': self.df_raw.groupby(['year', 'inst'])['mid'].count().max()},
            {'instruments': "_", 'observations': self.df_raw.groupby(['year'])['mid'].count().median(), 'observations per year': self.df_raw.groupby(['year', 'inst'])['mid'].count().median()}

        ]

        df_descstats = pd.DataFrame(summary, index=['overall', 'minimum per instrument', 'maximum per instrument', 'median per instrument']).T
        return df_descstats

    def run_backtest(self, data: str | pd.DataFrame, range_short=(24, 264, 24), range_long=(120, 800, 120), range_signal=(24, 120, 48) , sample_size=0.8, sample=None):
        """
        Run a brute-force grid search backtest for a MACD strategy on a given instrument.

        Parameters
        ----------
        data : str or pd.DataFrame
            Instrument data as a DataFrame or the name of the instrument file to load.
            If a string is provided, it should match the filename in the data folder without the extension.
        range_short : iterable
            Range of values to test for the short EMA window. Defaults to (24, 264, 24).
        range_long : iterable
            Range of values to test for the long EMA window. Defaults to (120, 800, 120).
        range_signal : iterable
            Range of values to test for the signal EMA window. Defaults to (24, 120, 48).
        sample_size : int
            Number of rows to use from the data for the test sample (subset of full instrument data).
        sample : str or None, optional
            Indicates which data sample to use: 'in' for in-sample, 'out' for out-of-sample, or None for full sample.

        Returns
        -------
        pd.DataFrame
            DataFrame containing strategy statistics for each parameter combination, including Sharpe ratio and position sizing.
            Includes columns: 'params', 'inst', and backtest metrics.
        """

        # Extract the instrument name from the DataFrame
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, str):
            df = self.df_inst_raw(data)

        inst = df.inst.unique()[0]
        # Initialize the MACD strategy object for the instrument with input data
        macd = MACD(inst=inst, df=df, col_backtest='mid', col_predict='mid', backtest_sample=sample_size)

        # Print out which sample is being optimized for traceability
        print(f'running {sample} sample brute optimizer for {macd} {inst}')

        # Run the brute-force grid search on the defined parameter ranges
        macd.brute2(
            range_short=range_short,
            range_long=range_long,
            range_signal=range_signal,
            opt_stat='sharpe_mean_annualized',  # Use annualized Sharpe ratio as optimization criterion
            sample=sample                       # Specify in/out/full sample
        )

        # Retrieve the appropriate metrics object based on the sample type
        if sample == 'in':
            metrics = macd.metrics_IN
        elif sample == 'out':
            metrics = macd.metrics_OUT
        else:
            metrics = macd.metrics

        # Get the full strategy performance metrics as a DataFrame
        df_stats = metrics.df_strats_all()

        # Extract numeric values from strategy parameter names and store as string
        df_stats['params'] = df_stats.index.map(lambda x: str(re.findall(r'\d+', x)))

        # Add instrument name to the results for traceability
        df_stats['inst'] = inst

        # Return the full metrics table with strategy evaluation results
        return df_stats
    
    def run_random_backtest(self, inst, range_short=(24, 264, 24) ,range_long=(120, 800, 120), range_signal=(24, 120, 48), sample_size=0.8, n=10, seed_=43):
        """
        Run a randomized backtest across multiple in-sample slices for a single instrument using a MACD strategy.

        This function repeatedly samples random time-contiguous windows from the instrument's data,
        performs a brute-force grid search on each slice, and aggregates the results. The method is used
        to assess parameter robustness by observing their performance across diverse market conditions.

        Parameters
        ----------
        inst : str
            Instrument name to run the backtest on.
        range_short : iterable
            Range of short EMA window lengths to test.
        range_long : iterable
            Range of long EMA window lengths to test.
        range_signal : iterable
            Range of signal EMA window lengths to test.
        sample_size : int
            The number of rows to include in each random sample.
        n : int, optional
            The number of random samples to generate. Default is 10.
        seed_ : int, optional
            Random seed for reproducibility. Default is 43.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame containing backtest results for all random samples, including strategy metrics
            and parameter identifiers.
        """
        # Get the instrument identifier from the data
        if isinstance(inst, pd.DataFrame):
            df = inst
        elif isinstance(inst, str):
            # Load the instrument data from the raw DataFrame
            df = self.df_inst_raw(inst)

        # Initialize an empty list to collect DataFrames from each run
        rand = []

        # Initialize a counter for tracking fold number
        counter = 0

        # Loop through n randomly generated index ranges
        for i in rand_samples(df=df, n=n, sample_size=sample_size, seed_=seed_):
            counter += 1
            beg, end = i  # Get start and end indices of the random window

            # Log the range used for this sample
            print(f'rand sample {inst}: {counter}, beg: {beg}, end: {end}')

            # Extract the random time slice of the instrument's data
            df_rand = df.iloc[beg:end]

            # Run the MACD grid search on the selected sample
            df_stats = self.run_backtest(df_rand, range_short, range_long, range_signal, sample_size, sample=None)

            # Append the result to the list
            rand.append(df_stats)

        # Concatenate all result DataFrames into one
        df_rand = pd.concat(rand)

        # Return the aggregated results
        return df_rand

    def run_backtest_all(self, range_short=(24, 264, 24), range_long=(120, 800, 120), range_signal=(24, 120, 48), sample_size=0.8, n=10)->pd.DataFrame:
        """
        Full Backtest Execution Loop for All Instruments

        <p>
        This block of code executes the entire backtesting pipeline for all instruments in the dataset. 
        <br>
        <br> 
        For each instrument:
        <ol> 
        <li> Loads the historical data, 
        <li> Defines a grid of MACD hyperparameters, and 
        <li> Runs a brute-force grid search to evaluate strategy performance on three data partitions: <b>in-sample</b> (80%), <b>out-of-sample</b> (20%), and <b>full-sample</b> (100%). 
        <br> <br>
        <b> <u> Grid search: </u></b> 
        The function implements a brute force grid search over a specified range of MACD parameters (short EMA, long EMA, and signal EMA) for a given instrument. The strategy logic is encapsulated in the <code>MACD</code> class, which handles the instantiation, backtesting, and metric extraction process. The backtest results, including the Sharpe ratio and other statistics for each parameter combination, are compiled and returned as a DataFrame. Each result is tagged with the instrument and the corresponding parameter set. Each result is saved into an HDF5 file using an appropriate key (e.g., 'in', 'out', 'all').
        </p>

        <b> <u> Randomized In-Sample Grid Search: </u> </b> 
        This function extends the brute-force MACD grid search by applying it across <b>multiple randomized in-sample subsets</b> of a given instrument's data. It simulates variability in training data by randomly selecting multiple (e.g., 10) time-based folds from the in-sample period. For each random sample, the MACD grid search is performed using the <code>run_backtest</code> function, and the results are stored. The goal of this approach is to evaluate the stability and robustness of MACD parameter configurations across different market environments, using pooled performance metrics such as the aggregated Sharpe ratio. This technique helps minimize overfitting to a single slice of data and supports more generalizable parameter selection.

        <li>Saves the results of the backtest into an HDF file with under the appropriate key: 'in' for the insample backtest, 'out' for the out of sample backtest and 'rand' for the 'randomized' backtest.
        </p>

        Parameters
        ----------
        range_short : iterable
            Range of values to test for the short EMA window.
        range_long : iterable
            Range of values to test for the long EMA window.
        range_signal : iterable
            Range of values to test for the signal EMA window.
        sample_size : int
            Number of rows to use from the data for the test sample (subset of full instrument data).
        n : int, optional
            Number of random samples to generate for randomized in-sample backtests. Default is 10.

        Returns
        -------
        pd.DataFrame
            DataFrame containing strategy statistics for each instrument and parameter combination,
            including Sharpe ratio and position sizing.
        """
        # Print the total number of grid combinations being tested
        print(f'length of grid: {len(grid(range_short, range_long, range_signal))}')


        insts_all = self.insts_all
        print(f'Number of instruments: {len(insts_all)}')
        print(f'Instruments: {insts_all}')

        # Loop through each instrument in the full list

        counter = 1
        for inst in insts_all:
                print(f'processing instrument {counter}/{len(insts_all)}: {inst}')
                path_data = self.path_data(inst)
                path_backtest = self.path_backtest(inst)

                # Run in-sample grid search and save results under 'in' key
                df_stats = self.run_backtest(data=inst, range_short=range_short, range_long=range_long, range_signal=range_signal, sample_size=sample_size, sample='in')
                
                # Save results to appropriate key in a new HDF5 file (mode='w' creates a new file)
                # df_stats.to_hdf(path_backtest, key='in', mode='w')

                # Run backtest on both out-of-sample (20%) and full sample (100%)
                for s in ['out', None]:
                    df_stats = self.run_backtest(
                        data=inst,
                        range_short=range_short,
                        range_long=range_long,
                        range_signal=range_signal,
                        sample_size=sample_size,
                        sample=s
                    )
                    # Append results to appropriate key in the same HDF5 file (mode='a' appends to existing file)
                    if s is None:
                        df_stats.to_hdf(path_backtest, key='all', mode='a')  # Full-sample
                    else:
                        df_stats.to_hdf(path_backtest, key=s, mode='a')      # Out-of-sample

                # Run randomized in-sample backtests (e.g., 10 folds of 80% of in-sample)
                df_rand = self.run_random_backtest(
                    inst=inst,
                    range_short=range_short,
                    range_long=range_long,
                    range_signal=range_signal,
                    sample_size=sample_size,
                    n=n
                )
                # Save the randomized results to the same HDF5 file under 'rand' key
                df_rand.to_hdf(path_backtest, key='rand', mode='a')

                # Increment instrument counter
                counter += 1


        return None  # No return value needed, results are saved to disk
    
    def get_stats(self, inst):
        """
        Retrieve and compare in-sample vs. out-of-sample backtest metrics for a given instrument.

        This function loads backtest results for a specific instrument from disk, calculates
        Sharpe ratios, assigns ranks, and compares in- and out-of-sample performance using
        both absolute deltas and a two-sample t-test.

        Parameters
        ----------
        inst : str
            The instrument identifier (used to locate its saved backtest file).
        top : int, optional
            Reserved for future use (e.g., returning only the top N parameter sets).

        Returns
        -------
        df_stats_inst_compare : pd.DataFrame
            A multi-index DataFrame comparing in-sample and out-of-sample statistics for each
            parameter configuration. Includes Sharpe ratio deltas, t-test statistics, and p-values.
        inst : str
            The instrument identifier, returned for convenience.
        """

        # Construct file path to the backtest results for the instrument
        # path_backtest = path.join(folder_main, folder_backtest, f'{inst}.h5')
        path_backtest = self.path_backtest(inst)

        # Metrics to extract and compare
        cols_metrics = ['returns_mean', 'std', 'sharpe', 'rank_sharpe']
        stats = []

        # Load and process results for both 'in' and 'out' samples
        for s in ['in', 'out']:
            df_stats = pd.read_hdf(path_backtest, key=s)  # Load HDF5 table
            df_stats['sample'] = s

            # Compute Sharpe ratio manually
            df_stats['sharpe'] = df_stats['returns_mean'] / df_stats['std']

            # Rank by Sharpe ratio (lower rank is better)
            df_stats.sort_values('sharpe', ascending=True, inplace=True)
            df_stats.reset_index(drop=False, inplace=True)
            df_stats.rename(columns={'index': 'id'}, inplace=True)
            df_stats['rank_sharpe'] = df_stats.index + 1

            # Collect in/out sample data
            stats.append(df_stats)

        # Combine the stats into a single DataFrame
        df_stats_inst = pd.concat(stats)

        # Pivot so we can compare in vs out per parameter
        df_stats_inst_compare = df_stats_inst.pivot(index='params', columns='sample', values=cols_metrics + ['total_obs'])

        # Rank by in-sample Sharpe ratio (descending, best first)
        df_stats_inst_compare.sort_values(('rank_sharpe', 'in'), inplace=True, ascending=False)

        # Calculate absolute differences between in and out for each metric
        for metric in cols_metrics:
            df_stats_inst_compare[(metric, 'diff')] = (
                df_stats_inst_compare[(metric, 'in')] - df_stats_inst_compare[(metric, 'out')]
            ).abs()

        # Ensure metrics are consistently ordered by column
        df_stats_inst_compare.sort_index(axis=1, inplace=True)

        # Calculate standard error for the difference in means
        df_stats_inst_compare[('se', 'tstats')] = np.sqrt(
            (df_stats_inst_compare['std', 'in'] / df_stats_inst_compare['total_obs', 'in']) +
            (df_stats_inst_compare['std', 'out'] / df_stats_inst_compare['total_obs', 'out'])
        )

        # Compute t-statistic for difference in returns
        df_stats_inst_compare[('t', 'tstats')] = (
            df_stats_inst_compare[('returns_mean', 'diff')] / df_stats_inst_compare[('se', 'tstats')]
        )

        # Degrees of freedom for two-sample t-test
        df_stats_inst_compare[('df', 'tstats')] = (
            df_stats_inst_compare['total_obs', 'in'] + df_stats_inst_compare['total_obs', 'out'] - 2
        )

        # Compute two-tailed p-value from t-statistic
        df_stats_inst_compare[('p', 'tstats')] = 2 * (
            1 - t.cdf(np.abs(df_stats_inst_compare[('t', 'tstats')]),
                    df_stats_inst_compare[('df', 'tstats')])
        )

        # Round p-values for presentation
        df_stats_inst_compare[('p', 'tstats')] = df_stats_inst_compare[('p', 'tstats')].apply(lambda x: round(x, 2))

        # Final sort to ensure top Sharpe-ranked in-sample parameters are first
        df_stats_inst_compare.sort_values(('rank_sharpe', 'in'), ascending=False, inplace=True)
        df_stats_inst_compare.sort_index(axis=1, inplace=True)

        return df_stats_inst_compare, inst
    

    def get_stats_rand(self, inst):
        """
        Compute pooled Sharpe ratio and performance metrics from randomized in-sample backtests.

        This function reads the results of randomized grid searches for a specific instrument,
        aggregates metrics across the random folds, and computes the pooled Sharpe ratio for
        each parameter configuration using weighted statistics. Parameter sets are then ranked
        by their pooled Sharpe score.

        Parameters
        ----------
        inst : str
            The instrument identifier used to locate its saved randomized backtest results.

        Returns
        -------
        df_rand_calc : pd.DataFrame
            A DataFrame with one row per parameter configuration, including:
            - Weighted mean return and pooled standard deviation
            - Pooled Sharpe ratio
            - Rank based on pooled Sharpe
            - Aggregate statistics across folds
        """

        # Load randomized backtest results for the instrument
        path_backtest = self.path_backtest(inst)
        df_rand = pd.read_hdf(path_backtest, key='rand')

        # Compute individual Sharpe ratios per fold
        df_rand['sharpe'] = df_rand['returns_mean'] / df_rand['std']

        # Compute weighted return and unbiased variance for each fold
        df_rand['returns_mean_weighted'] = df_rand['returns_mean'] * df_rand['total_obs']
        df_rand['var'] = df_rand['std'] ** 2
        df_rand['var_weighted'] = df_rand['var'] * (df_rand['total_obs'] - 1)

        # Group by instrument and parameter set, then aggregate metrics across folds
        df_rand_calc = df_rand.groupby(['inst', 'params']).agg(
            returns_mean_mean=pd.NamedAgg(column="returns_mean", aggfunc="mean"),
            returns_mean_annualized_mean=pd.NamedAgg(column="returns_mean_annualized", aggfunc="mean"),
            sharpe_mean_annualized_mean=pd.NamedAgg(column="sharpe_mean_annualized", aggfunc="mean"),
            std_mean=pd.NamedAgg(column="std", aggfunc="mean"),
            std_annualized_mean_mean=pd.NamedAgg(column="std_annualized", aggfunc="mean"),
            n_groups=pd.NamedAgg(column="total_obs", aggfunc="count"),
            total_obs=pd.NamedAgg(column="total_obs", aggfunc="sum"),
            returns_mean_weighted_sum=pd.NamedAgg(column="returns_mean_weighted", aggfunc="sum"),
            var_weighted_sum=pd.NamedAgg(column="var_weighted", aggfunc="sum"),
        )

        # Compute weighted average return
        df_rand_calc['returns_mean_wa'] = df_rand_calc['returns_mean_weighted_sum'] / df_rand_calc['total_obs']

        # Compute pooled standard deviation using unbiased pooled variance
        df_rand_calc['std_pooled'] = np.sqrt(
            df_rand_calc['var_weighted_sum'] / (df_rand_calc['total_obs'] - df_rand_calc['n_groups'])
        )

        # Compute pooled Sharpe ratio
        df_rand_calc['sharpe_pooled'] = df_rand_calc['returns_mean_wa'] / df_rand_calc['std_pooled']

        # Sort by pooled Sharpe to assign ranking
        df_rand_calc.sort_values('sharpe_pooled', ascending=True, inplace=True)
        df_rand_calc.reset_index(drop=False, inplace=True)
        df_rand_calc.rename(columns={'index': 'id'}, inplace=True)
        df_rand_calc['rank_sharpe'] = df_rand_calc.index + 1

        # Re-sort in descending order (best performing first)
        df_rand_calc.sort_values('sharpe_pooled', ascending=False, inplace=True)

        return df_rand_calc

    def summarize_stats(self, inst, top_param: list):
        df_stats_inst_compare = self.get_stats(inst)[0]

        df_top_sharpe = df_stats_inst_compare.loc[df_stats_inst_compare.index.isin(top_param), :]

        stats_inst = {
            inst: {
                'params': str(df_top_sharpe.index.tolist()),

                # In-sample values
                'sharpe.in': df_top_sharpe[('sharpe', 'in')].values[0],
                'rank_sharpe.in': df_top_sharpe[('rank_sharpe', 'in')].values[0],
                'returns_mean.in': df_top_sharpe[('returns_mean', 'in')].values[0],

                # Out-of-sample values
                'sharpe.out': df_top_sharpe[('sharpe', 'out')].values[0],
                'rank_sharpe.out': df_top_sharpe[('rank_sharpe', 'out')].values[0],
                'returns_mean.out': df_top_sharpe[('returns_mean', 'out')].values[0],

                # Differences
                'sharpe.diff': df_top_sharpe[('sharpe', 'diff')].values[0],
                'rank_sharpe.diff': df_top_sharpe[('rank_sharpe', 'diff')].values[0],
                'returns_mean.diff': df_top_sharpe[('returns_mean', 'diff')].values[0],
                'returns_mean.p': df_top_sharpe[('p', 'tstats')].values[0],
            }
        }

        df_stats_top = pd.DataFrame(stats_inst).T
        return df_stats_top
    
    def df_top_inout(self, top: int=1, rand=True):

        """
        <h3>Top In/Out Sample Strategy Evaluation</h3>
        <p>
        This function evaluates the top-performing MACD strategy parameters for each instrument based on in-sample Sharpe ratio. It retrieves the best parameter set for each instrument and computes in-sample and out-of-sample performance metrics, including Sharpe ratios, returns, and rank deltas. The results are aggregated into a single DataFrame for cross-instrument analysis.
        </p>

        <h4>1. Evaluate Top In-Sample Strategy Across All Instruments</h4>

        <p>
        This function identifies the best-performing MACD parameter set (based on in-sample Sharpe ratio) for each instrument and evaluates how well that top parameter generalizes to out-of-sample data. For each instrument, it uses the <code>get_stats</code> function to retrieve in/out sample comparison metrics and selects the top-ranked parameter set based on in-sample Sharpe. Then, using <code>summarize_stats</code>, it computes performance deltas and rank correlation metrics (e.g., Spearman, Kendall) to quantify the consistency of that top parameter's performance.
        </p>

        <p>
        All summary statistics are aggregated into a single DataFrame across instruments, allowing for cross-sectional analysis of model robustness and generalization quality. The final DataFrame includes measures such as Sharpe ratio differences, rank consistency, and statistical test p-values.
        </p>

        <h4>2. Evaluation of Top Pooled Sharpe Strategies from Randomized In-Sample Folds</h4>

        <p>
        This block identifies and evaluates the best-performing MACD parameter set for each instrument based on the <b>pooled Sharpe ratio</b> computed across multiple randomized in-sample folds. The goal is to determine whether the parameter set that performs most consistently across different random in-sample windows also generalizes well to out-of-sample data.
        </p>

        <p>
        For each instrument, the function <code>get_stats_rand</code> retrieves aggregated results from 10 randomized folds. The parameter set with the highest pooled Sharpe ratio is selected, and its performance is summarized using <code>summarize_stats</code>, which computes in/out-sample performance, absolute differences in key metrics, and a p-value for the return difference. The final output is a summary table across all instruments.
        </p>

        Parameters:
        ----------
        top : int, optional
            The number of top parameter sets to evaluate for each instrument. Default is 1.
        
        rand : bool, optional
            If True, evaluates the top parameter set based on pooled Sharpe ratio from randomized folds.
            If False, evaluates the top parameter set based on in-sample Sharpe ratio.
            Default is True.
            
        Returns:
        -------
        df_top_in_out_rand : pd.DataFrame
            A DataFrame containing the top in/out sample performance metrics for each instrument.

        """
        # Initialize an empty list to collect summary results for all instruments
        top_in_out_rand = []

        # Loop through each instrument in the list
        for inst in self.insts_all:

            # Load pooled Sharpe performance metrics from randomized folds
            # df_stats_rand = get_stats_rand(inst)

            # Select the parameter set with the highest pooled Sharpe ratio
            if rand:
                top_param = self.get_stats_rand(inst).head(top).params.tolist()
            else:
                top_param = self.get_stats(inst)[0].head(top).index.tolist()

            # Evaluate the selected top parameter using in/out-sample statistics
            df_stats_rand = self.summarize_stats(inst, top_param)

            # Append the result to the collection list
            top_in_out_rand.append(df_stats_rand)

        # Concatenate all individual summaries into a single DataFrame
        df_top_in_out_rand = pd.concat(top_in_out_rand)

        # Convert all numeric columns to float for compatibility (except 'params')
        for c in df_top_in_out_rand.columns:
            if c != 'params':
                df_top_in_out_rand[c] = df_top_in_out_rand[c].astype('float')

        ordered_cols = [
                    'params',
                    'sharpe.in', 'sharpe.out',
                    'rank_sharpe.in', 'rank_sharpe.out',
                    'returns_mean.in', 'returns_mean.out',
                    'sharpe.diff', 'rank_sharpe.diff', 'returns_mean.diff', 'returns_mean.p'
                ]

        return df_top_in_out_rand[ordered_cols]
    
    def compare(self, df_top_inout=None, df_top_in_out_rand=None):
        # Create comparison DataFrame
        if df_top_inout is None:
            df_top_inout = self.df_top_inout(1, False)
        if df_top_in_out_rand is None:
            df_top_in_out_rand = self.df_top_inout(1, True)
        # Ensure both DataFrames have the same index
        df_compare = pd.DataFrame({
            'Simple In/Out Sample splits: Abs(rank delta)': df_top_inout['rank_sharpe.diff'],
            'Randomized In/Out Sample splits: Abs(rank delta)': df_top_in_out_rand['rank_sharpe.diff'],
            'In-Sample Rank Diff': df_top_inout['returns_mean.diff'],
            'Randomized Pooled Rank Diff': df_top_in_out_rand['returns_mean.diff']
            
        })

        df_compare['Randomized In/Out Sample splits: Abs(returns delta)'] = df_top_in_out_rand['returns_mean.diff']
        df_compare['Simple In/Out Sample splits: Abs(returns delta)'] = df_top_inout['returns_mean.diff']

        df_compare.index.name = 'instrument'


        compare_overall = df_compare.mean(axis=0)
        compare_overall.name = 'Overall average'
        df_compare = pd.concat([df_compare, compare_overall.to_frame().T])


        df_compare.loc[df_compare['Randomized In/Out Sample splits: Abs(rank delta)'] < df_compare['Simple In/Out Sample splits: Abs(rank delta)'], 'Improved Ranking'] = True
        df_compare.loc[~(df_compare['Randomized In/Out Sample splits: Abs(rank delta)'] < df_compare['Simple In/Out Sample splits: Abs(rank delta)']), 'Improved Ranking'] = False

        df_compare.loc[df_compare['Randomized In/Out Sample splits: Abs(returns delta)'] < df_compare['Simple In/Out Sample splits: Abs(returns delta)'], 'Improved Abs Returns Delta'] = True
        df_compare.loc[~(df_compare['Randomized In/Out Sample splits: Abs(returns delta)'] < df_compare['Simple In/Out Sample splits: Abs(returns delta)']), 'Improved Abs Returns Delta'] = False

        return df_compare
    

    def plot_compare(self, df_compare=None):
        if df_compare is None:
            df_compare = self.compare()

        # Plot
        ax = df_compare[['Simple In/Out Sample splits: Abs(rank delta)',	'Randomized In/Out Sample splits: Abs(rank delta)']].plot(kind='bar', figsize=(16, 6))
        ax.set_title('Rank Difference Comparison: Simple In/Out Sample vs. Randomized In Sample (Pooled Sharpe)', fontsize=14)
        ax.set_ylabel('Absolute Rank Difference')
        ax.set_xlabel('Instrument')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Plot
        ax = df_compare[['Simple In/Out Sample splits: Abs(returns delta)', 'Randomized In/Out Sample splits: Abs(returns delta)']].plot(kind='bar', figsize=(16, 6))
        ax.set_title('Abs Mean Returns Delta: Simple In/Out Sample vs. Randomized In Sample (Pooled Sharpe)', fontsize=14)
        ax.set_ylabel('Absolute Mean Returns Difference')
        ax.set_xlabel('Instrument')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def ttest(self, df_top_in_out_rand = None, df_top_inout = None):
        if df_top_inout is None:
            df_top_inout = self.df_top_inout(1, False)
        if df_top_in_out_rand is None:
            df_top_in_out_rand = self.df_top_inout(1, True)

        # --- T-test for absolute Sharpe rank difference ---

        # Perform a one-sided Welch's t-test comparing the rank difference between the randomized and simple splits.
        # The test checks whether the randomized sampling leads to significantly smaller absolute changes in Sharpe-based ranks.
        # 'equal_var=False' uses Welch’s correction for unequal variances.
        # 'alternative="less"' means the test checks if the randomized sample has a lower mean than the simple sample.
        t_stat_rank, p_value_rank = ttest_ind(
            df_top_in_out_rand['rank_sharpe.diff'],  # Sample A: Rank differences from randomized in-sample evaluation
            df_top_inout['rank_sharpe.diff'],        # Sample B: Rank differences from fixed in/out split evaluation
            equal_var=False,
            alternative='less'  # One-sided: Is Sample A < Sample B?
        )

        # Display the test results for ranking stability
        print(f"t-statistic Ranking: {t_stat_rank:.4f}")
        print(f"p-value Ranking: {p_value_rank:.4f}")

        # --- T-test for absolute mean return difference ---

        # Same structure as above, now evaluating whether the randomized sampling yields smaller differences in mean return.
        # Results are used to test the hypothesis that pooled randomized evaluation is more stable in returns as well.
        t_stat_mean, p_value_mean = ttest_ind(
            df_top_in_out_rand['returns_mean.diff'],  # Sample A: Mean return differences from randomized sampling
            df_top_inout['returns_mean.diff'],        # Sample B: Mean return differences from fixed split
            equal_var=False,
            alternative='less'  # One-sided test: Does Sample A have lower mean than Sample B?
        )

        # Display the test results for mean return stability
        print(f"t-statistic Mean: {t_stat_mean:.4f}")
        print(f"p-value Mean: {p_value_mean:.4f}")

        # --- Summary Table ---

        # Construct a summary DataFrame that presents:
        # 1. The average absolute rank and return deltas for both sampling methods
        # 2. The corresponding t-statistics and p-values from the two tests above
        # Note: Return deltas are scaled to basis points (bps) for interpretability
        df_tstat = pd.DataFrame(data={
            'Abs (Rank delta)': [
                df_top_in_out_rand['rank_sharpe.diff'].mean(),  # Randomized average
                df_top_inout['rank_sharpe.diff'].mean(),        # Simple split average
                t_stat_rank,                                    # t-statistic
                p_value_rank                                    # p-value
            ],
            'Abs (Mean Returns delta) - bps': [
                df_top_in_out_rand['returns_mean.diff'].mean() * 10000,  # Randomized in bps
                df_top_inout['returns_mean.diff'].mean() * 10000,        # Simple split in bps
                t_stat_mean,                                             # t-statistic
                p_value_mean                                             # p-value
            ],
        },
        index=[
            'Randomized In/Out Sample splits', 
            'Simple In/Out Sample splits', 
            'tstat', 
            'pvalue'
        ])

        return df_tstat
