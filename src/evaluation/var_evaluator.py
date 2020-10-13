from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from gluonts.evaluation import Evaluator
import scipy.stats as stats

from src.evaluation.helpers import get_seasonal_errors, get_mase

class VarEvaluator(object):

    def __init__(
        self, 
        var_results: List[VARResultsWrapper],
        train_datasets: List[np.ndarray],
        original_datasets: List[np.ndarray],
        initial_log_values: List[np.ndarray],
        horizon: int = 6,
        freq: str = 'M',
        var_diff: bool = False,
    ) -> None:

        self.var_results = var_results
        self.lag_orders = [results.k_ar for results in var_results]
        self.horizon = horizon
        self.train_datasets = train_datasets
        self.original_datasets = [
            np.ma.masked_invalid(original_dataset)
            for original_dataset in original_datasets
        ] 
        self.initial_log_values = initial_log_values
        self.evaluator = Evaluator()
        self.freq = freq
        self.var_diff = var_diff

    def evaluate(self) -> np.ndarray:
        self.forecasts = self._forecasts()
        if len(self.forecasts) > 1:
            self.forecasts = [
                f[:,0] for f in self.forecasts
            ]
            self.targets = [
                test[-self.horizon:, 0]
                for test in self.original_datasets
            ]
            self.past_datas = [
                test[:-self.horizon, 0]
                for test in self.original_datasets
            ]
        else:
            self.targets = [
                test[-self.horizon:, :]
                for test in self.original_datasets
            ]
            self.past_datas = [
                test[:-self.horizon, :]
                for test in self.original_datasets
            ]

    def mase(self) -> float:
        seasonal_errors = get_seasonal_errors(self.past_datas, self.freq)
        mase = [
            get_mase(t, f, s) for t, f, s in zip(
                self.targets, self.forecasts, seasonal_errors
            )
        ]
        return np.mean(mase)

    def mape(self) -> float:
        mape = [
            self.evaluator.mape(t, f) for t, f in zip(
                self.targets, self.forecasts
            )
        ]
        return np.mean(mape)

    def smape(self) -> float:
        smape = [
            self.evaluator.smape(t, f) for t, f in zip(
                self.targets, self.forecasts
            )
        ]
        return np.mean(smape)

    def top_coefficients(
        self,
        grp_names: List[str],
        cov_names: List[str],
        pct_obs: List[float],
        pval_min: float = 0.05
    ) -> None:
        
        avg_coefs = [[] for i in range(self.var_results[0].coefs.shape[-1])]
        for result in self.var_results:
            pvalues = self._robust_pvalues(result) 
            sig_idxs = pvalues[:,0] < pval_min
            coefs = result.coefs[:,0,:].flatten()
            for idx in np.where(sig_idxs)[0]:
                i = idx % len(avg_coefs)
                avg_coefs[i].append(coefs[i])
        avg_coefs = np.array([np.mean(coefs) for coefs in avg_coefs])
        avg_coefs = np.nan_to_num(avg_coefs)

        if not len(cov_names):
            names = grp_names
            
            plt.clf()
            plt.scatter(avg_coefs, pct_obs)
            plt.xlabel('Avg Stat. Sig. Coefs.')
            plt.ylabel('Pct. Obs.')
            plt.savefig(f'plots/var_coefs_obs_corr.png')

        else:
            names = ['malnutrition series'] + cov_names
        
        top_idxs = np.flip(np.argsort(np.abs(avg_coefs)))
        for idx in top_idxs:
            print(f'{names[idx]}: {round(avg_coefs[idx], 2)}')
        
    def _robust_pvalues(self, result) -> np.ndarray:
        try:
            pvalues = result.pvalues[1:]
        except np.linalg.LinAlgError:
            z = result.endog_lagged
            cov_params = np.kron(np.linalg.pinv(z.T @ z), result.sigma_u)
            stderr = np.sqrt(np.diag(cov_params))
            stderr = stderr.reshape((result.df_model, result.neqs), order='C')
            tvalues = result.params / stderr
            pvalues = 2 * stats.norm.sf(np.abs(tvalues))
        
        return pvalues

    def plot(self, start: Timestamp, num_viz: int = 10) -> None:
        
        fig, ax = plt.subplots(
            num_viz, 
            1, 
            sharex=True, 
            sharey=True, 
            figsize=(10, 7)
        )

        if len(self.forecasts) == 1:
            original_datasets = self.original_datasets[0].T
            forecasts = self.forecasts[0].T
        else:
            original_datasets = self.original_datasets
            forecasts = self.forecasts

        for i, (series, fcast) in enumerate(zip(
            original_datasets,
            forecasts
        )):
            if i == num_viz:
                break

            # plot original, uninterpolated series
            index = pd.date_range(
                start=start,
                freq=self.freq,
                periods=series.shape[0]
            )
            
            if series.ndim > 1:
                series = series[:,0]
            ts = pd.Series(series, index=index)
            ts.plot(ax=ax[i], legend=False)
            
            # plot mean forecast
            fcast_index = pd.date_range(
                start=index[-self.horizon],
                freq=self.freq,
                periods=fcast.shape[0]
            )
            fpoint = pd.Series(fcast, index=fcast_index)
            fpoint.plot(color='g', ax=ax[i], legend=False)
            
        plt.savefig(f'plots/visualize_forecast_var.png')

    def _forecasts(self):
        forecast_logs = [
            results.forecast(
                train[-lag:],
                self.horizon
            )
            for results, train, lag in zip(
                self.var_results, self.train_datasets, self.lag_orders
            )
        ]
        if self.var_diff:
            train_logs = [
                np.sum(train, axis=0) 
                for train in self.train_datasets
            ]
            train_logs = [
                t_logs + init_logs for t_logs, init_logs in zip(
                    train_logs, self.initial_log_values
                )
            ]
            forecast_logs = [
                np.cumsum(f_logs_diffed, axis = 1) 
                for f_logs_diffed in forecast_logs
            ]
            forecast_logs = [
                f_logs + t_logs[-1]
                for f_logs, t_logs in zip(
                    forecast_logs,
                    train_logs
                )
            ]
        forecasts = [np.expm1(f_logs) for f_logs in forecast_logs]
        return forecasts
