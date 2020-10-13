from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
from gluonts.dataset.common import ListDataset
from gluonts.evaluation._base import get_seasonality

from src.evaluation.helpers import get_seasonal_errors, get_mase

class GluonEvaluator(object):

    def __init__(
        self, 
        predictor: Predictor,
        test_dataset: ListDataset,
        original_dataset: ListDataset,
        multivariate: bool = True, 
        num_samples: int = 100,
    ) -> None:

        self.predictor = predictor
        self.test_dataset = test_dataset
        self.original_series = list(original_dataset)
        self.num_samples = num_samples
        self.agg_metrics = None
        self.ind_metrics = None

        original_values = [
            np.ma.masked_invalid(series['target'])
            for series in self.original_series
        ]

        if multivariate:
            if original_values[0].ndim > 1:
                self.evaluator = MultivariateEvaluator(eval_dims = [0])
                original_values = [s[0] for s in original_values]
            else:
                self.evaluator = MultivariateEvaluator()
        else:
            self.evaluator = Evaluator()

        self.targets = [
            series[-self.predictor.prediction_length:]
            for series in original_values
        ]
        self.past_datas = [
            series[:-self.predictor.prediction_length]
            for series in original_values
        ]

    def evaluate(self) -> None:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.test_dataset,
            predictor=self.predictor, 
            num_samples=self.num_samples
        )
        self.agg_metrics, self.ind_metrics = self.evaluator(ts_it, forecast_it)
        seasonal_errors = get_seasonal_errors(self.past_datas)

    def mase(self) -> float:
        seasonal_errors = get_seasonal_errors(self.past_datas, self.predictor.freq)
        forecasts, _ = make_evaluation_predictions(
            dataset=self.test_dataset,
            predictor=self.predictor, 
            num_samples=self.num_samples
        )

        self.forecasts = list(forecasts)
        if self.forecasts[0].mean.ndim > 1:
            if len(self.forecasts) == 1:
                self.forecasts = self.forecasts[0].quantile(0.5).T
            else:
                self.forecasts = [f.quantile(0.5).T[0] for f in self.forecasts]
        else:
            self.forecasts = [f.quantile(0.5) for f in self.forecasts]   
        
        mase = [
            get_mase(t, f, s) for t, f, s in zip(
                self.targets, self.forecasts, seasonal_errors
            )
        ]

        return np.mean(mase)

    def mape(self) -> float:
        return self.agg_metrics['MAPE']

    def smape(self) -> float:
        return self.agg_metrics['sMAPE']

    def plot(
        self, 
        num_viz: int = 10, 
    ) -> None:
        
        forecasts, _ = make_evaluation_predictions(
            dataset=self.test_dataset,
            predictor=self.predictor, 
            num_samples=self.num_samples
        )

        fig, ax = plt.subplots(
            num_viz, 
            1, 
            sharex=True, 
            sharey=True, 
            figsize=(10, 7)
        )
        for i, (series, fcast) in enumerate(zip(
            self.original_series,
            self.forecasts, 
        )):

            if i == num_viz:
                break
            
            # plot original, uninterpolated series
            index = pd.date_range(
                start=series['start'],
                freq=self.predictor.freq,
                periods=series['target'].shape[-1]
            )

            if series['target'].ndim > 1:
                ts = pd.Series(series['target'][0], index=index)
            else:
                ts = pd.Series(series['target'], index=index)
            ts.plot(ax=ax[i], legend=False)

            # plot mean/median forecast
            if self.predictor.freq == 'W':
                offset = pd.DateOffset(weeks = self.predictor.prediction_length - 1)
            elif self.predictor.freq == 'M':
                offset = pd.DateOffset(months = self.predictor.prediction_length - 1)
            fcast_index = pd.date_range(
                start=index[-1] - offset,
                freq=self.predictor.freq,
                periods=self.predictor.prediction_length
            )

            fcast = pd.Series(fcast, index=fcast_index)
            fcast.plot(color='g', ax=ax[i], legend=False)
            
        plt.savefig(f'plots/visualize_forecast_gluon.png')

