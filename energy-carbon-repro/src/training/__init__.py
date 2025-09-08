"""训练优化模块"""

from .mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    AutomaticMixedPrecisionTrainer,
    MemoryOptimizer,
    create_mixed_precision_trainer,
    get_optimal_mixed_precision_config
)
from .onecycle_scheduler import (
    OneCycleLR,
    AdaptiveOneCycleLR,
    WarmupScheduler,
    LearningRateSchedulerManager,
    SuperConvergenceTrainer,
    create_onecycle_scheduler,
    estimate_optimal_max_lr
)

from .purged_validation import (
    TimeSeriesValidationConfig,
    PurgedTimeSeriesCV,
    TimeSeriesModelValidator,
    TimeSeriesBacktester,
    create_purged_cv,
    validate_timeseries_model
)

from .unified_framework import (
    UnifiedTrainingConfig,
    UnifiedTrainingFramework,
    BaseTrainingStrategy,
    MLTrainingStrategy,
    DLTrainingStrategy,
    create_unified_trainer,
    quick_train,
    benchmark_training_strategies
)

__all__ = [
    'MixedPrecisionConfig',
    'MixedPrecisionTrainer', 
    'AutomaticMixedPrecisionTrainer',
    'MemoryOptimizer',
    'create_mixed_precision_trainer',
    'get_optimal_mixed_precision_config',
    'OneCycleLR',
    'AdaptiveOneCycleLR',
    'WarmupScheduler',
    'LearningRateSchedulerManager',
    'SuperConvergenceTrainer',
    'create_onecycle_scheduler',
    'estimate_optimal_max_lr',
    'TimeSeriesValidationConfig',
    'PurgedTimeSeriesCV',
    'TimeSeriesModelValidator',
    'TimeSeriesBacktester',
    'create_purged_cv',
    'validate_timeseries_model',
    'UnifiedTrainingConfig',
    'UnifiedTrainingFramework',
    'BaseTrainingStrategy',
    'MLTrainingStrategy',
    'DLTrainingStrategy',
    'create_unified_trainer',
    'quick_train',
    'benchmark_training_strategies'
]