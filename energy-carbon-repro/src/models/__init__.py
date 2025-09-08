"""模型初始化模块"""

from .transformer_fusion import LNGTransformerFusion
from .mlr import MultipleLinearRegression
from .gpr import GaussianProcessRegression
from .hgbr_baseline import AdaptiveHGBRBaseline, create_adaptive_hgbr_baseline
from .adaptive_strategy import (
    SampleSizeAdaptiveModelFactory, 
    AdaptiveModelEnsemble,
    create_adaptive_model
)
from .residual_framework import (
    ResidualModelingFramework,
    LinearBaseline,
    ApproximateGPR,
    create_residual_model
)
from .tcn_attention import (
    OptimizedTCNModel,
    TCNLinearAttention,
    LinearAttention,
    TemporalConvNet,
    create_tcn_model
)

__all__ = [
    'LNGTransformerFusion',
    'MultipleLinearRegression', 
    'GaussianProcessRegression',
    'AdaptiveHGBRBaseline',
    'create_adaptive_hgbr_baseline',
    'SampleSizeAdaptiveModelFactory',
    'AdaptiveModelEnsemble',
    'create_adaptive_model',
    'ResidualModelingFramework',
    'LinearBaseline', 
    'ApproximateGPR',
    'create_residual_model',
    'OptimizedTCNModel',
    'TCNLinearAttention',
    'LinearAttention',
    'TemporalConvNet',
    'create_tcn_model'
]