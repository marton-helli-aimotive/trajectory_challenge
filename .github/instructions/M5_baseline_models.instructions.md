# Milestone 5: Baseline Models Implementation

## Overview
Implement baseline trajectory prediction models (Constant Velocity, Constant Acceleration, Polynomial Regression) to establish performance benchmarks and validate the complete pipeline from data to predictions.

## Duration: 4-5 days

## Objectives
- Implement simple baseline models for trajectory prediction
- Validate end-to-end pipeline functionality
- Establish performance benchmarks for comparison
- Create comprehensive testing framework
- Generate initial model comparison results

## Dependencies
- **M4**: Model interface and training framework
- **M3**: Feature engineering pipeline
- **M2**: Data pipeline providing trajectory data

## Acceptance Criteria

### 1. Constant Velocity Model ✅
```python
class ConstantVelocityPredictor(AbstractTrajectoryPredictor):
    """Predicts future trajectory assuming constant velocity"""
    
    def fit(self, X: FeatureMatrix, y: TrajectoryTargets) -> None:
        # No training required - uses last known velocity
        pass
    
    def predict(self, X: FeatureMatrix, horizon: float) -> PredictionResult:
        # Extrapolate using last velocity vector
        # position(t) = position(t0) + velocity * (t - t0)
```

**Features**:
- Uses last observed velocity vector
- Linear extrapolation for future positions
- No training required (zero-parameter model)
- Provides uncertainty based on velocity variance

### 2. Constant Acceleration Model ✅
```python
class ConstantAccelerationPredictor(AbstractTrajectoryPredictor):
    """Predicts trajectory assuming constant acceleration"""
    
    def predict(self, X: FeatureMatrix, horizon: float) -> PredictionResult:
        # position(t) = position(t0) + velocity*t + 0.5*acceleration*t²
```

**Features**:
- Uses last observed acceleration vector
- Quadratic extrapolation for future positions
- Estimates acceleration from recent trajectory points
- Better for scenarios with consistent acceleration patterns

### 3. Polynomial Regression Model ✅
```python
class PolynomialTrajectoryPredictor(AbstractTrajectoryPredictor):
    """Polynomial regression on trajectory features"""
    
    def __init__(self, degree: int = 3, regularization: float = 0.01):
        self.degree = degree
        self.x_regressor = PolynomialRegression(degree, regularization)
        self.y_regressor = PolynomialRegression(degree, regularization)
```

**Features**:
- Separate polynomial models for x and y coordinates
- Engineered features: position, velocity, acceleration, jerk
- Ridge regularization to prevent overfitting
- Cross-validation for degree selection

### 4. Enhanced Feature Engineering for Baselines ✅
**Temporal Features**:
- Position history (last 5-10 time steps)
- Velocity smoothing using moving averages
- Acceleration estimation from finite differences
- Jerk calculation for higher-order dynamics

**Physics-Based Features**:
- Speed magnitude and direction
- Heading angle and rate of change
- Centripetal acceleration (for curved paths)
- Distance traveled in recent time window

### 5. Baseline Model Testing ✅
**Unit Tests**:
- Model creation and parameter validation
- Prediction output format verification
- Edge case handling (single point, stationary vehicle)
- Uncertainty estimation correctness

**Integration Tests**:
- End-to-end pipeline with real NGSIM data
- Cross-validation with multiple data splits
- Performance benchmarking across scenarios
- Memory and runtime profiling

## Technical Deliverables

1. **ConstantVelocityPredictor** with uncertainty quantification
2. **ConstantAccelerationPredictor** with adaptive acceleration estimation
3. **PolynomialTrajectoryPredictor** with regularization and feature engineering
4. **BaselineModelEvaluator** for comprehensive performance assessment
5. **Benchmark dataset** with ground truth for all model types
6. **Performance comparison report** establishing baseline metrics

## Model Implementation Details

### Constant Velocity Implementation
```python
def predict(self, X: FeatureMatrix, horizon: float) -> PredictionResult:
    # Extract last position and velocity
    last_position = X[:, -2:]  # Last [x, y]
    last_velocity = X[:, -4:-2]  # Last [vx, vy]
    
    # Generate time steps for prediction horizon
    dt = 0.1  # 100ms time steps
    time_steps = np.arange(dt, horizon + dt, dt)
    
    # Linear extrapolation: position = p0 + v * t
    predictions = []
    for t in time_steps:
        future_pos = last_position + last_velocity * t
        predictions.append(future_pos)
    
    return PredictionResult(
        predicted_positions=np.array(predictions),
        timestamps=time_steps,
        confidence_scores=self._compute_confidence(last_velocity)
    )
```

### Uncertainty Quantification
- **Constant Velocity**: Uncertainty grows linearly with time
- **Constant Acceleration**: Uncertainty grows quadratically
- **Polynomial**: Bootstrap confidence intervals

### Performance Optimization
- **Vectorized Operations**: NumPy for batch processing
- **Efficient Feature Extraction**: Minimal computation overhead
- **Memory Management**: Streaming for large datasets
- **Parallel Processing**: Multi-core prediction for large batches

## Evaluation Metrics (Baseline)

### Primary Metrics
- **Average Displacement Error (ADE)**: Mean position error over trajectory
- **Final Displacement Error (FDE)**: End-point prediction accuracy
- **Root Mean Square Error (RMSE)**: Overall trajectory accuracy

### Secondary Metrics
- **Minimum Distance**: Closest approach safety metric
- **Lateral Error**: Cross-track deviation
- **Heading Error**: Direction prediction accuracy

### Performance Benchmarks
Target performance for baseline models on NGSIM data:
- **1-second horizon**: ADE < 2.0m, FDE < 3.0m
- **3-second horizon**: ADE < 5.0m, FDE < 8.0m
- **5-second horizon**: ADE < 8.0m, FDE < 12.0m

## Dependencies & Integration
- **NumPy/SciPy**: Mathematical operations and optimization
- **scikit-learn**: Polynomial regression and cross-validation
- **Matplotlib/Plotly**: Visualization of predictions vs ground truth

## Risks & Mitigations
- **Risk**: Poor baseline performance indicating data issues
- **Mitigation**: Comprehensive data quality validation
- **Risk**: Overfitting in polynomial model
- **Mitigation**: Proper regularization and cross-validation

## Success Criteria
- [ ] All three baseline models implemented and tested
- [ ] Models achieve target performance benchmarks
- [ ] End-to-end pipeline processes NGSIM data successfully
- [ ] Uncertainty estimates are calibrated and meaningful
- [ ] Test coverage >90% for all model components
- [ ] Performance comparison shows expected model ranking

## Testing Strategy

### Model-Specific Tests
```python
def test_constant_velocity_prediction():
    # Test linear extrapolation correctness
    # Test edge cases (zero velocity, single point)
    # Test uncertainty grows linearly with time

def test_polynomial_regression_training():
    # Test model fitting with synthetic data
    # Test regularization effect on overfitting
    # Test cross-validation parameter selection
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    # Load NGSIM data → Feature extraction → Model training → Prediction
    # Verify complete pipeline works with real data
    # Check performance metrics are reasonable
```

## Notes
- Focus on correctness and simplicity over optimization
- Ensure models provide meaningful uncertainty estimates
- Use baseline results to validate evaluation metrics
- Document any data quality issues discovered during implementation
