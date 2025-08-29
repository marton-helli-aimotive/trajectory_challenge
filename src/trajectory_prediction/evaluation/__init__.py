"""
Comprehensive evaluation framework for trajectory prediction models.

This module provides:
- Safety-critical metrics (TTC, minimum distance, collision risk)
- Standard ML metrics (RMSE, MAE, ADE, FDE)
- Probabilistic evaluation metrics
- Cross-validation and statistical testing
- Model comparison and benchmarking
"""

from .metrics import (
    SafetyMetrics,
    TrajectoryMetrics,
    ProbabilisticMetrics,
    MetricCalculator
)

from .evaluator import (
    ModelEvaluator,
    CrossValidator,
    ModelComparator
)

from .statistical import (
    StatisticalTester,
    SignificanceTest
)

from .comparison import (
    ModelComparisonInterface,
    ABTestingFramework,
    ComparisonReport
)

from .benchmarking import (
    PerformanceBenchmarker,
    BenchmarkResult,
    BenchmarkReport
)

__all__ = [
    "SafetyMetrics",
    "TrajectoryMetrics", 
    "ProbabilisticMetrics",
    "MetricCalculator",
    "ModelEvaluator",
    "CrossValidator",
    "ModelComparator",
    "StatisticalTester",
    "SignificanceTest",
    "ModelComparisonInterface",
    "ABTestingFramework",
    "ComparisonReport",
    "PerformanceBenchmarker",
    "BenchmarkResult",
    "BenchmarkReport"
]