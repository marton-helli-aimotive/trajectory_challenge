"""Feature engineering and validation components for vehicle trajectory prediction."""

from .extraction import (
    TrajectoryFeatureExtractor,
    VelocityFeatureExtractor,
    AccelerationFeatureExtractor,
    CurvatureFeatureExtractor,
    LaneChangeFeatureExtractor,
    SpatialTemporalFeatureExtractor,
    ContextualFeatureExtractor,
)
from .quality import (
    TrajectoryQualityMetrics,
    CompletenessMetrics,
    SmoothnessMetrics,
    ConsistencyMetrics,
    PhysicsConstraintValidator,
)
from .augmentation import (
    TrajectoryAugmentor,
    NoiseInjectionAugmentor,
    InterpolationAugmentor,
    SyntheticScenarioGenerator,
    AdversarialExampleGenerator,
)
from .store import (
    FeatureStore,
    FeatureDefinition,
    FeatureVersion,
    FeatureCache,
)
from .validation import (
    FeatureValidator,
    PhysicsInformedValidator,
    FeatureQualityScorer,
    ValidationReport,
)

__all__ = [
    # Feature Extraction
    "TrajectoryFeatureExtractor",
    "VelocityFeatureExtractor", 
    "AccelerationFeatureExtractor",
    "CurvatureFeatureExtractor",
    "LaneChangeFeatureExtractor",
    "SpatialTemporalFeatureExtractor",
    "ContextualFeatureExtractor",
    
    # Quality Metrics
    "TrajectoryQualityMetrics",
    "CompletenessMetrics",
    "SmoothnessMetrics", 
    "ConsistencyMetrics",
    "PhysicsConstraintValidator",
    
    # Data Augmentation
    "TrajectoryAugmentor",
    "NoiseInjectionAugmentor",
    "InterpolationAugmentor",
    "SyntheticScenarioGenerator",
    "AdversarialExampleGenerator",
    
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureVersion",
    "FeatureCache",
    
    # Validation
    "FeatureValidator",
    "PhysicsInformedValidator",
    "FeatureQualityScorer",
    "ValidationReport",
]