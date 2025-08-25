---
mode: agent
---
Refactor the codebase. Separate fuctions and types to different modules:
- ngsim: data loading and cacheing. It shouldn't make any data transform, just load the data and returns a single DataFrame
- trajectory: data processing and transformation functions, trajectory detection and handling
- gui: Streamlit-based user interface components for data exploration and visualization.
- utils: utility functions and types used across the other modules

Generate a clear API for each module, including function signatures, expected inputs and outputs. Keep it clear, concise, focus on usability and maintainability.