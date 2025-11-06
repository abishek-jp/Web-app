import streamlit as st
import joblib
import pickle
import os
import sys
from pathlib import Path

st.title("ML Pipeline Diagnostic Tool")

# Display environment info
st.subheader("Environment Information")
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")

# Check if file exists and get info
file_path = 'full_pipeline.pkl'
st.subheader("File Analysis")

if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    st.success(f"✓ File found: {file_path}")
    st.write(f"File size: {file_size} bytes")
    
    # Try multiple loading methods
    st.subheader("Loading Attempts")
    
    # Method 1: Standard joblib
    st.write("**Attempt 1: Standard joblib.load**")
    try:
        with st.spinner("Loading with standard joblib..."):
            pipeline1 = joblib.load(file_path)
        st.success("✓ SUCCESS: Standard joblib.load worked!")
        st.write(f"Pipeline type: {type(pipeline1)}")
    except Exception as e:
        st.error(f"✗ FAILED: {e}")
    
    # Method 2: Joblib with mmap_mode=None
    st.write("**Attempt 2: joblib.load with mmap_mode=None**")
    try:
        with st.spinner("Loading with mmap_mode=None..."):
            pipeline2 = joblib.load(file_path, mmap_mode=None)
        st.success("✓ SUCCESS: Joblib with mmap_mode=None worked!")
        st.write(f"Pipeline type: {type(pipeline2)}")
    except Exception as e:
        st.error(f"✗ FAILED: {e}")
    
    # Method 3: Direct pickle
    st.write("**Attempt 3: Direct pickle.load**")
    try:
        with st.spinner("Loading with pickle..."):
            with open(file_path, 'rb') as f:
                pipeline3 = pickle.load(f)
        st.success("✓ SUCCESS: Direct pickle.load worked!")
        st.write(f"Pipeline type: {type(pipeline3)}")
    except Exception as e:
        st.error(f"✗ FAILED: {e}")
    
    # Method 4: Check file contents
    st.write("**Attempt 4: File content analysis**")
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(100)  # Read first 100 bytes
        st.write(f"First 20 bytes (hex): {first_bytes[:20].hex()}")
        
        # Try to identify pickle protocol
        if first_bytes[0:2] == b'\x80':  # Pickle protocol indicator
            protocol = first_bytes[1] if len(first_bytes) > 1 else 'unknown'
            st.write(f"Pickle protocol version: {protocol}")
            
    except Exception as e:
        st.error(f"File analysis failed: {e}")

else:
    st.error(f"✗ File not found: {file_path}")
    st.info("Looking for file in common locations...")
    
    # Check common paths
    possible_paths = [
        './full_pipeline.pkl',
        '../full_pipeline.pkl', 
        './models/full_pipeline.pkl',
        '../models/full_pipeline.pkl',
        Path(__file__).parent / 'full_pipeline.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.success(f"Found file at: {path}")
            break
    else:
        st.error("File not found in any common locations")

# Library versions check
st.subheader("Library Versions")
try:
    import sklearn
    import pandas as pd
    import numpy as np
    
    st.write(f"scikit-learn: {sklearn.__version__}")
    st.write(f"pandas: {pd.__version__}")
    st.write(f"numpy: {np.__version__}")
    st.write(f"joblib: {joblib.__version__}")
except Exception as e:
    st.error(f"Version check failed: {e}")
