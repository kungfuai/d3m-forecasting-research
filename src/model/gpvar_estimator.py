from typing import List, Optional

import numpy as np
from gluonts.core.component import DType
from gluonts.distribution import DistributionOutput
from gluonts.distribution.lowrank_gp import LowrankGPOutput
from gluonts.core.component import validated
from gluonts.time_feature import TimeFeature
from gluonts.trainer import Trainer
from gluonts.transform import cdf_to_gaussian_forward_transform
from gluonts.model.gpvar._estimator import GPVAREstimator, get_lags_for_frequency

from src.model.deepvar_estimator import time_features_from_frequency_str

class GPVAREstimatorPatch(GPVAREstimator):

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        # number of dimension to sample at training time
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        target_dim_sample: Optional[int] = None,
        distr_output: Optional[DistributionOutput] = None,
        rank: Optional[int] = 2,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        shuffle_target_dim: bool = True,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 100,
        use_marginal_transformation: bool = False,
        dtype: DType = np.float32,
    ) -> None:

        self.trainer = trainer
        self.dtype = dtype

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_layers > 0, "The value of `num_layers` should be > 0"
        assert num_cells > 0, "The value of `num_cells` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_eval_samples` should be > 0"
        assert dropout_rate >= 0, "The value of `dropout_rate` should be >= 0"

        if distr_output is not None:
            self.distr_output = distr_output
        else:
            self.distr_output = LowrankGPOutput(rank=rank)
        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.target_dim_sample = (
            target_dim
            if target_dim_sample is None
            else min(target_dim_sample, target_dim)
        )
        self.shuffle_target_dim = shuffle_target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else get_lags_for_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else time_features_from_frequency_str(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling
        self.conditioning_length = conditioning_length
        self.use_marginal_transformation = use_marginal_transformation
        if self.use_marginal_transformation:
            self.output_transform = cdf_to_gaussian_forward_transform
        else:
            self.output_transform = None