#!/usr/bin/env python3
"""
Simple Streamlit test to verify basic functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(
        page_title="Simple Test Dashboard",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Simple Test Dashboard")
    
    st.write("This is a simple test to verify Streamlit is working.")
    
    # Test basic components
    st.header("Basic Components Test")
    
    # Test sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Home", "📊 Data", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        st.subheader("Home Page")
        st.write("Welcome to the test dashboard!")
        
        # Test button
        if st.button("Click Me!"):
            st.success("Button clicked successfully!")
        
        # Test slider
        value = st.slider("Select a value", 0, 100, 50)
        st.write(f"Selected value: {value}")
        
    elif page == "📊 Data":
        st.subheader("Data Page")
        
        # Test dataframe
        df = pd.DataFrame({
            'A': np.random.randn(10),
            'B': np.random.randn(10),
            'C': np.random.randn(10)
        })
        st.dataframe(df)
        
        # Test chart
        st.line_chart(df)
        
    elif page == "⚙️ Settings":
        st.subheader("Settings Page")
        
        # Test checkboxes
        option1 = st.checkbox("Option 1", value=True)
        option2 = st.checkbox("Option 2", value=False)
        
        st.write(f"Option 1: {option1}")
        st.write(f"Option 2: {option2}")
        
        # Test selectbox
        choice = st.selectbox("Choose an option", ["A", "B", "C"])
        st.write(f"Selected: {choice}")

if __name__ == "__main__":
    main()