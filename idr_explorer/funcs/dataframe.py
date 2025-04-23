import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import ijson
import glob
from tqdm import tqdm
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


def retrieve_dataframe_stream(metadata_directory, use_streamlit=True):

    # Initialize lists for data extraction
    nb_images_per_experiment = []
    studies_types = []
    samples_types = []
    metadata_filenames = []
    experiment_ids = []
    imaging_methods = []
    total_nb_images = 0
    cell_experiments = 0
    tissue_experiments = 0

    if use_streamlit:
        # Initialize progress bar
        progress_bar = st.progress(0)

        # Retrieve information from metadata files in the directory
        for i, filename in enumerate(os.listdir(metadata_directory)):
            if filename.startswith("metadata_experiment_") and filename.endswith(
                ".json"
            ):
                with open(os.path.join(metadata_directory, filename), "rb") as f:

                    # Counter for total number of images check
                    total_nb_images_in_file = 0

                    # Create an ijson parser for the current file
                    parser = ijson.parse(f)

                    # Initialize variables
                    screen = None
                    project = None

                    # Iterate through the parser events
                    for prefix, event, value in parser:
                        if event == "map_key":
                            # Store the current key to determine the data to extract
                            if value == "screens":
                                # Start of screens section, create a new screen dictionary
                                screen = {}

                            elif value == "projects":
                                # Start of projects section, create a new project dictionary
                                project = {}

                            else:
                                current_key = value

                        elif event == "number":

                            if current_key == "total_images_in_screen":
                                # Store the number of images in the current screen
                                nb_images_per_experiment.append(value)
                                total_nb_images += value
                                total_nb_images_in_file += value

                            elif current_key == "total_images_in_project":
                                # Store the number of images in the current project
                                nb_images_per_experiment.append(value)
                                total_nb_images += value
                                total_nb_images_in_file += value
                            

                            elif current_key == "total_images":
                                assert total_nb_images_in_file == value

                        elif event == "string":
                            if current_key == "Sample Type":
                                # Store the sample type
                                if value == "cell":
                                    cell_experiments += 1
                                    samples_types.append("cell")
                                elif value == "tissue":
                                    tissue_experiments += 1
                                    samples_types.append("tissue")

                            elif current_key == "Study Type":
                                # Store the study type
                                studies_types.append(value)

                            elif current_key == "Imaging Method":
                                imaging_methods.append(value)

                        screen = None
                        project = None

                    progress = (i + 1) / len(os.listdir(metadata_directory))
                    progress_bar.progress(progress)

                metadata_filenames.append(filename)

    elif use_streamlit is False:
        # Retrieve information from metadata files in the directory
        for filename in tqdm(os.listdir(metadata_directory)):
            if filename.startswith("metadata_experiment_") and filename.endswith(
                ".json"
            ):
                with open(os.path.join(metadata_directory, filename), "rb") as f:

                    # Counter for total number of images check
                    total_nb_images_in_file = 0

                    # Create an ijson parser for the current file
                    parser = ijson.parse(f)

                    # Initialize variables
                    screen = None
                    project = None

                    # Iterate through the parser events
                    for prefix, event, value in parser:
                        if event == "map_key":
                            # Store the current key to determine the data to extract
                            if value == "screens":
                                # Start of screens section, create a new screen dictionary
                                screen = {}

                            elif value == "projects":
                                # Start of projects section, create a new project dictionary
                                project = {}

                            else:
                                current_key = value

                        elif event == "number":

                            if current_key == "total_images_in_screen":
                                # Store the number of images in the current screen
                                nb_images_per_experiment.append(value)
                                total_nb_images += value
                                total_nb_images_in_file += value
                                metadata_filenames.append(filename)
                                experiment_ids.append('screen_' + filename.split('_')[2].split('.')[0])

                            elif current_key == "total_images_in_project":
                                # Store the number of images in the current project
                                nb_images_per_experiment.append(value)
                                total_nb_images += value
                                total_nb_images_in_file += value
                                metadata_filenames.append(filename)
                                experiment_ids.append('project_' + filename.split('_')[2].split('.')[0])

                            elif current_key == "total_images":
                                assert total_nb_images_in_file == value

                        elif event == "string":
                            if current_key == "Sample Type":
                                # Store the sample type
                                if value == "cell":
                                    cell_experiments += 1
                                    samples_types.append("cell")
                                elif value == "tissue":
                                    tissue_experiments += 1
                                    samples_types.append("tissue")

                            elif current_key == "Study Type":
                                # Store the study type
                                studies_types.append(value)

                            elif current_key == "Imaging Method":
                                imaging_methods.append(value)

                        screen = None
                        project = None


    # Create the DataFrame
    df = pd.DataFrame(
        {
            "Select": [False] * len(nb_images_per_experiment),
            "sample_type": samples_types,
            "study_type": studies_types,
            "imaging_method": imaging_methods,
            "metadata_file": metadata_filenames,
            "experiment_id": experiment_ids,
            "total_nb_images": nb_images_per_experiment,
        }
    )

    return df


def display_dataframe(dataframe):
    gd = GridOptionsBuilder.from_dataframe(dataframe)
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd.build()

    grid_table = AgGrid(
        dataframe,
        height=200,
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
    )

    return dataframe, grid_table


@st.cache_data()
def filter_dataframe(dataframe, threshold):

    dataframe["nb_images_selected"] = threshold

    # # Calculate the total size of the files selected using split and retrieve the first element
    # dataframe["total_size_thresholded"] = (
    #     dataframe["nb_files_selected"] * dataframe["average_file_size (MB)"].split().str[0].astype(float)
    # )

    return dataframe[dataframe["total_nb_images"] >= threshold]


def get_npz_file_size(dataframe, directory):

    # Create empty lists to store the file sizes
    file_sizes = []

    # Iterate over the metadata files
    for experiment_id in dataframe["experiment_id"]:

        if experiment_id is not None:
            
            # Construct the corresponding images_experiment file name
            images_file = f"experiment_{experiment_id}"

            # Construct the file pattern with a wildcard
            file_pattern = os.path.join(directory, images_file + "*.npz")

            # Find all files matching the pattern
            file_list = glob.glob(file_pattern)

            if file_list:
                # Get the size of the file in bytes
                file_size = os.path.getsize(file_list[0])
                # Convert the file size to MB
                file_size_mb = file_size / (1024 * 1024)
                average_file_size = file_size_mb / 10

            else:
                # If the file doesn't exist, set the file size to NA
                average_file_size = np.nan
        else:
            # If no number is found in the metadata file name, set the file size to NA
            average_file_size = np.nan

        # Append the file size to the list
        file_sizes.append(average_file_size)

    return file_sizes

def get_tiff_file_size(dataframe, directory):

    # Create empty lists to store the file sizes
    file_sizes = []

    # Iterate over the metadata files
    for experiment_id in dataframe["experiment_id"]:

        if experiment_id is not None:
            
            # Construct the corresponding images_experiment file name
            images_file = f"experiment_{experiment_id}"

            # Construct the file pattern with a wildcard
            file_pattern = os.path.join(directory, images_file + "*.tiff")

            # Find all files matching the pattern
            file_list = glob.glob(file_pattern)

            if file_list:
                # Get the size of the file in bytes
                file_size = os.path.getsize(file_list[0])
                # Convert the file size to MB
                file_size_mb = file_size / (1024 * 1024)
                average_file_size = file_size_mb / 10

            else:
                # If the file doesn't exist, set the file size to NA
                average_file_size = np.nan
        else:
            # If no number is found in the metadata file name, set the file size to NA
            average_file_size = np.nan

        # Append the file size to the list
        file_sizes.append(average_file_size)

    return file_sizes

def get_png_file_size(dataframe, directory):
    """
    Get the size of the png files in the directory
    """

    # Create empty lists to store the file sizes
    average_file_sizes = []

    # Iterate over the metadata files
    for experiment_id in dataframe["experiment_id"]:

        if experiment_id is not None:
            
            tmp_file_sizes = []

            # Construct the corresponding images_experiment file name
            images_file = f"experiment_{experiment_id}"

            # Construct the file pattern with a wildcard
            file_pattern = os.path.join(directory, images_file + "*.png")

            # Find all files matching the pattern
            file_list = glob.glob(file_pattern)

            for img in file_list:
                # Get the size of the file in bytes
                file_size = os.path.getsize(img)
                # Convert the file size to MB
                file_size_mb = file_size / (1024 * 1024)
                tmp_file_sizes.append(file_size_mb)

            average_file_size = np.mean(tmp_file_sizes)
            average_file_sizes.append(average_file_size)

        else:
            # If no number is found in the metadata file name, set the file size to NA
            average_file_size = np.nan
            average_file_sizes.append(average_file_size)


    return average_file_sizes

def convert_file_size(file_size):
    """
    Converts a file size in bytes to a human-readable format.
    
    Args:
        file_size (int): Size of the file in bytes.
    
    Returns:
        str: Human-readable file size with appropriate unit.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while file_size >= 1024 and index < len(units) - 1:
        file_size /= 1024.0
        index += 1
    return f"{file_size:.2f} {units[index]}"
