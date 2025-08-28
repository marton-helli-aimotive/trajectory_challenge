#!/usr/bin/env python3
"""
Minimal dashboard for testing.
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(
        page_title="Minimal Dashboard",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Vehicle Trajectory Prediction Dashboard")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Overview", "📊 Trajectory Visualization", "🔍 Model Comparison"]
    )
    
    if page == "🏠 Overview":
        st.markdown("## 📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Available", 6)
        
        with col2:
            st.metric("Sample Trajectories", 5)
        
        with col3:
            st.metric("Prediction Horizon", "10s")
        
        with col4:
            st.metric("Update Frequency", "Real-time")
        
        st.success("✅ Dashboard is working!")
        
    elif page == "📊 Trajectory Visualization":
        st.markdown("## 📊 Trajectory Visualization")
        
        # Simple trajectory data
        t = np.linspace(0, 10, 50)
        x = t * 10 + np.random.normal(0, 1, 50)
        y = t * 5 + np.random.normal(0, 1, 50)
        
        # Create dataframe
        df = pd.DataFrame({
            'Time': t,
            'X': x,
            'Y': y
        })
        
        st.line_chart(df.set_index('Time')[['X', 'Y']])
        
    elif page == "🔍 Model Comparison":
        st.markdown("## 🔍 Model Comparison")
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': ['Constant Velocity', 'Polynomial Regression', 'KNN'],
            'RMSE': [1.2, 0.8, 0.9],
            'ADE': [0.9, 0.6, 0.7],
            'FDE': [2.1, 1.5, 1.8]
        })
        
        st.dataframe(performance_data)
        
        if st.button("Run Comparison"):
            st.success("Model comparison completed!")

if __name__ == "__main__":
    main()