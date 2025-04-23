import json
import os
import streamlit as st

# Function to load metadata from a file
@st.cache_data
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load metadata from each file and append to a list
@st.cache_data
def append_metadata(metadata_files, directory):
    metadata = []
    progress_bar = st.progress(0)
    for i, file in enumerate(metadata_files):
        metadata.append(load_metadata(os.path.join(directory, file)))
        progress = (i + 1) / len(metadata_files)
        progress_bar.progress(progress)
    
    return metadata