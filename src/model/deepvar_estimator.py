from typing import List, Optional

import numpy as np
from pandas.tseries.frequencies import to_offset
from gluonts.core.component import DType, validated
from gluonts.time_feature import TimeFeature
from gluonts.model.deepvar._estimator import (
    FourierDateFeatures, 
    DeepVAREstimator,
    get_lags_for_frequency
)
from gluonts.trainer import Trainer
from gluonts.distribution import (
    DistributionOutput,
    LowrankMultivariateGaussianOutput,
)

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    offset = to_offset(freq_str)
    multiple, granularity = offset.n, offset.name

    features = {
        "M": ["weekofyear"],
        "W-SUN": ["daysinmonth", "weekofyear"],
        "D": ["dayofweek"],
        "B": ["dayofweek", "dayofyear"],
        "H": ["hour", "dayofweek"],
        "min": ["minute", "hour", "dayofweek"],
        "T": ["minute", "hour", "dayofweek"],
    }

    assert granularity in features, f"freq {granularity} not supported"

    feature_classes: List[TimeFeature] = [
        FourierDateFeatures(freq=freq) for freq in features[granularity]
    ]
    return feature_classes

class DeepVAREstimatorPatch(DeepVAREstimator):

    def __init__(
        self,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "lstm",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        distr_output: Optional[DistributionOutput] = None,
        rank: Optional[int] = 5,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        conditioning_length: int = 200,
        use_marginal_transformation=False,
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
        assert all(
            [c > 0 for c in cardinality]
        ), "Elements of `cardinality` should be > 0"
        assert (
            embedding_dimension > 0
        ), "The value of `embedding_dimension` should be > 0"

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )

        if distr_output is not None:
            self.distr_output = distr_output
        else:
            self.distr_output = LowrankMultivariateGaussianOutput(
                dim=target_dim, rank=rank
            )

        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.conditioning_length = conditioning_length
        self.use_marginal_transformation = use_marginal_transformation

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

        if self.use_marginal_transformation:
            self.output_transform: Optional[
                Callable
            ] = cdf_to_gaussian_forward_transform
        else:
            self.output_transform = None