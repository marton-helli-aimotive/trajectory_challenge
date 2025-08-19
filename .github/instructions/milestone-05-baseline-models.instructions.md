# Milestone 5: Baseline & Classical Prediction Models

## Objective
Implement baseline and classical machine learning models for trajectory prediction to establish performance benchmarks and validate the modeling pipeline.

## Success Criteria
- [ ] 4 baseline/classical models implemented with unified interface
- [ ] Model factory pattern for easy model selection
- [ ] Hyperparameter optimization for each model
- [ ] Cross-validation pipeline for temporal data
- [ ] Initial evaluation metrics implementation

## Technical Requirements

### 1. Model Implementations
```python
# Baseline Models
1. Constant Velocity (CV): Linear extrapolation
2. Constant Acceleration (CA): Quadratic extrapolation

# Classical ML Models  
3. Polynomial Regression: with engineered features
4. K-Nearest Neighbors: trajectory similarity with DTW
```

### 2. Model Architecture
- **Abstract Model Interface**: Unified predict/fit/evaluate methods
- **Model Factory Pattern**: Configuration-driven model creation
- **Model Registry**: Centralized model management and versioning
- **Pipeline Integration**: Seamless feature pipeline connection

### 3. Training Infrastructure
- **Time-Series Cross-Validation**: Respecting temporal ordering
- **Hyperparameter Optimization**: Using Optuna for automated tuning
- **Model Persistence**: Serialization and loading mechanisms
- **Training Monitoring**: Progress tracking and early stopping

### 4. Initial Evaluation Metrics
- **RMSE**: Root Mean Square Error for trajectory accuracy
- **ADE**: Average Displacement Error over prediction horizon
- **FDE**: Final Displacement Error at prediction end
- **Computational Performance**: Training time and inference speed

### 5. Model-Specific Features
- **KNN Similarity**: Dynamic Time Warping for trajectory matching
- **Polynomial Features**: Position, velocity, acceleration, jerk terms
- **Temporal Features**: Time-based predictors for motion patterns

## Dependencies
- Milestone 4: Feature Engineering Framework

## Risk Factors
- **Low Risk**: Well-established classical models with known behavior
- **Medium Risk**: KNN with DTW may have performance issues on large datasets
- **Mitigation**: Implement efficient DTW algorithms and indexing
- **Mitigation**: Start with small dataset subsets for validation

## Deliverables
1. 4 implemented prediction models with unified interface
2. Model factory and registry system
3. Hyperparameter optimization pipeline
4. Time-series cross-validation framework
5. Initial evaluation metrics and benchmarks
6. Model performance analysis and comparison

## Next Milestone
Milestone 6: Advanced ML Models (depends on this milestone)
