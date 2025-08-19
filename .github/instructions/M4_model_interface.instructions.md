# Milestone 4: Model Interface & Training Framework

## Overview
Design and implement a unified interface for all trajectory prediction models, establish training/evaluation framework, and set up MLOps infrastructure for experiment tracking and model management.

## Duration: 3-4 days

## Objectives
- Create unified model interface supporting all prediction approaches
- Implement training and evaluation framework with cross-validation
- Set up MLflow for experiment tracking and model registry
- Establish model selection and hyperparameter optimization
- Design ensemble framework for model combination

## Dependencies
- **M3**: Feature engineering providing feature vectors
- **M1**: Core architecture and patterns

## Acceptance Criteria

### 1. Unified Model Interface ✅
```python
class AbstractTrajectoryPredictor(ABC):
    @abstractmethod
    def fit(self, X: FeatureMatrix, y: TrajectoryTargets) -> None:
        """Train the model on historical trajectory data"""
    
    @abstractmethod
    def predict(self, X: FeatureMatrix, horizon: float) -> PredictionResult:
        """Predict future trajectory points"""
    
    @abstractmethod
    def predict_uncertainty(self, X: FeatureMatrix) -> UncertaintyResult:
        """Provide prediction uncertainty estimates"""
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return model metadata and hyperparameters"""
```

**Prediction Result Schema**:
```python
class PredictionResult(BaseModel):
    predicted_positions: np.ndarray  # Shape: (n_samples, horizon_steps, 2)
    timestamps: np.ndarray          # Future timestamps
    confidence_scores: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = {}

class UncertaintyResult(BaseModel):
    mean_predictions: np.ndarray
    variance_estimates: np.ndarray
    confidence_intervals: np.ndarray  # [lower, upper] bounds
    uncertainty_type: str  # 'aleatoric', 'epistemic', 'total'
```

### 2. Training Framework ✅
**Cross-Validation Strategy**:
- **Time-Series Split**: Respecting temporal order
- **Vehicle-Based Split**: Preventing data leakage
- **Stratified Split**: Balancing scenario types
- **Online Learning**: Incremental model updates

**Training Pipeline**:
- **Data preprocessing** and feature scaling
- **Hyperparameter optimization** using Optuna
- **Model validation** with multiple metrics
- **Early stopping** and regularization
- **Model persistence** and checkpointing

### 3. MLOps Infrastructure ✅
**MLflow Integration**:
- **Experiment tracking** with parameters, metrics, artifacts
- **Model registry** with versioning and stage management
- **Model serving** with REST API endpoints
- **Model comparison** and selection tools

**Experiment Management**:
- **Hyperparameter logging** for reproducibility
- **Artifact storage** for models, plots, reports
- **Metric tracking** across training epochs
- **Model lineage** tracking data and code versions

### 4. Model Factory Pattern ✅
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: DictConfig) -> AbstractTrajectoryPredictor:
        """Factory method for creating model instances"""
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of available model types"""
    
    @classmethod
    def create_ensemble(cls, models: List[str], weights: Optional[List[float]] = None) -> EnsembleModel:
        """Create ensemble of multiple models"""
```

### 5. Evaluation Framework Foundation ✅
**Model Evaluation Interface**:
```python
class ModelEvaluator:
    def evaluate(self, model: AbstractTrajectoryPredictor, 
                test_data: TrajectoryDataset) -> EvaluationResult
    
    def compare_models(self, models: List[AbstractTrajectoryPredictor],
                      test_data: TrajectoryDataset) -> ComparisonResult
    
    def statistical_significance_test(self, results_a: EvaluationResult,
                                    results_b: EvaluationResult) -> StatTestResult
```

## Technical Deliverables

1. **AbstractTrajectoryPredictor** base class with complete interface
2. **TrainingPipeline** with cross-validation and hyperparameter optimization
3. **ModelFactory** supporting all planned model types
4. **MLflowExperimentTracker** for logging experiments
5. **EnsembleModel** for combining multiple predictors
6. **ModelRegistry** for model versioning and deployment

## Training Data Preparation

### Input Feature Design
- **Historical Trajectory**: Variable-length sequences (5-30 seconds)
- **Contextual Features**: Traffic, road geometry, weather
- **Interaction Features**: Surrounding vehicle states
- **Temporal Features**: Time-based patterns

### Target Design
- **Prediction Horizons**: 1s, 3s, 5s, 10s future positions
- **Multi-Step Prediction**: Sequence of future positions
- **Probabilistic Targets**: Distribution parameters for uncertainty models
- **Safety-Critical Events**: Lane changes, emergency braking

### Data Splitting Strategy
```python
class TrajectoryDataSplitter:
    def temporal_split(self, ratio: Tuple[float, float, float]) -> Tuple[Dataset, Dataset, Dataset]
    def vehicle_split(self, ratio: Tuple[float, float, float]) -> Tuple[Dataset, Dataset, Dataset]
    def scenario_stratified_split(self, scenarios: List[str]) -> Tuple[Dataset, Dataset, Dataset]
```

## Configuration Management

### Model Configurations
```yaml
models:
  baseline_cv:
    type: "constant_velocity"
    parameters: {}
  
  polynomial_regression:
    type: "polynomial"
    degree: 3
    regularization: 0.01
  
  gaussian_process:
    type: "gaussian_process"
    kernel: "rbf"
    noise_level: 0.1
```

### Training Configurations
```yaml
training:
  cross_validation:
    strategy: "time_series"
    n_splits: 5
    test_size: 0.2
  
  optimization:
    optimizer: "optuna"
    n_trials: 100
    objective: "ade"
```

## Performance Requirements
- **Model Training**: Complete within 30 minutes for baseline models
- **Hyperparameter Optimization**: <2 hours for complex models
- **Model Inference**: <100ms per prediction
- **Memory Usage**: <4GB during training

## Dependencies & Integration
- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **scikit-learn**: Model selection and validation
- **Hydra**: Configuration management

## Risks & Mitigations
- **Risk**: Complex model interface hindering implementation
- **Mitigation**: Start simple, iterate based on actual model needs
- **Risk**: MLflow setup complexity
- **Mitigation**: Use simple local setup initially, scale later

## Success Criteria
- [ ] Unified interface supports all planned model types
- [ ] Training pipeline handles cross-validation correctly
- [ ] MLflow tracks experiments with full reproducibility
- [ ] Hyperparameter optimization finds better configs
- [ ] Ensemble models combine predictions effectively
- [ ] Model factory creates instances of all model types

## Notes
- Design interface to be model-agnostic (classical ML + deep learning)
- Ensure all models support uncertainty quantification
- Keep training pipeline flexible for online learning
- Plan for distributed training in future iterations
