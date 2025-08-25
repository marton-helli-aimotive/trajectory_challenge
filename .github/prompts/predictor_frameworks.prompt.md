---
mode: agent
---

Create a framework for exploring trajectory predictor algorithms. A trajectory predictor gets a trajectory and a timestamp as input and predicts the future trajectory of the object. The framework should allow users to:

1. **Select a Trajectory Predictor**: Users can choose from a list of available trajectory predictors (e.g., linear regression, LSTM, etc.).

2. **Configure Predictor Parameters**: Users can adjust the parameters of the selected predictor (e.g., learning rate, number of layers, etc.).

3. **Select trajectory**: Users can select the trajectory they want to analyze.

4. **Visualize Predictions**: The framework should provide tools for visualizing the predicted trajectories alongside the ground truth.

5. **Evaluate Performance**: Users can evaluate the performance of the predictor using metrics such as RMSE, MAE, etc.

6. **Save and Share Configurations**: Users can save their configurations and share them with others.

The framework should be built using Streamlit for the frontend and should be modular to allow for easy addition of new predictors and evaluation metrics.
