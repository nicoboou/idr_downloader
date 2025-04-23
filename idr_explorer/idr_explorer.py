import random
import os
import asyncio
import streamlit as st
import pandas as pd

# Local imports
from funcs.load import append_metadata
from funcs.dataframe import retrieve_dataframe, retrieve_dataframe_stream, display_dataframe, filter_dataframe, format_file_size
from screens.display_detailed_view import display_detailed_view
from screens.display_data_distribution import display_data_distribution
from screens.display_data_size import display_data_size
from screens.display_shannon_entropy import display_shannon_entropy

# Base URL for the IDR
IDR_BASE_URL = 'https://idr.openmicroscopy.org'

# ------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------- App Starts Here ------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------- #
# srry for inconvenience, but Streamlit doesn't like classes ..

# Create the Streamlit application
st.title('Metadata Explorer')

# Display a dropdown selection for the directory
experiment_type = st.selectbox('Select a type', ['cell', 'tissue'])

# Set the directory path based on the selection
directory = os.path.join('data', 'IDR', experiment_type, 'meta')

# Get a list of all JSON files in the selected directory
#metadata_files = [file for file in os.listdir(directory) if file.startswith('metadata_experiment_5') and file.endswith('.json')]

# Sort the file list by numerical order
#metadata_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

# Load metadata from each file and append to a list
#metadata = append_metadata(metadata_files, directory)

# Retrieve a dataframe from the full metadata
dataframe = retrieve_dataframe_stream(metadata_directory=directory)

# Display sidebar selection box to choose a file
select_action = st.sidebar.selectbox('Select an action', ['Overview', 'Detailed View'], index=0, key='select_action')

# --------------------------------- Display the overview --------------------------------- #
if select_action == 'Overview':
    # Create a Streamlit slider to set the filter threshold
    threshold = st.slider("Filter threshold", min_value=0, max_value=dataframe['total_nb_images'].max(), value=dataframe['total_nb_images'].min(), step=100)

    filtered_dataframe = filter_dataframe(dataframe=dataframe, threshold=threshold)

    df, grid_table = display_dataframe(dataframe=filtered_dataframe)
    selected_rows = grid_table['selected_rows']

    if bool(selected_rows):
        df_selected_rows = pd.DataFrame(selected_rows)
        st.write(f'**Total Number of Images**: {df_selected_rows["nb_files_selected"].sum()}')
        st.write(f'**Total Dataset Size**: {format_file_size(df_selected_rows["total_size"].sum())}')
        display_shannon_entropy(dataframe=df_selected_rows)
        display_data_distribution(dataframe=df_selected_rows)
    else:
        st.write(f'**Total Number of Images**: {filtered_dataframe["nb_files_selected"].sum() - threshold}')
        st.write(f'**Total Dataset Size**: {format_file_size(filtered_dataframe["total_size"].sum())}')
        display_shannon_entropy(dataframe=filtered_dataframe)
        display_data_distribution(dataframe=filtered_dataframe)

# --------------------------------- Displau the detailed view --------------------------------- #
elif select_action == 'Detailed View':
    # Display the overview
    display_detailed_view(metadata, metadata_files)
