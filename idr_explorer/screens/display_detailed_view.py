import random
import os
import asyncio
import streamlit as st

# Local imports
from funcs.display import display_images, get_screen_image_ids, get_project_image_ids

def display_detailed_view(metadata, metadata_files):
    # Display a sidebar selection box to choose a file
    selected_file = st.selectbox('Select a file', metadata_files, index=0, key='selected_file')

    # Display specific metadata from the selected file
    selected_metadata = metadata[metadata_files.index(selected_file)]

    # Cache image_ids
    screen_image_ids = get_screen_image_ids(selected_metadata)
    project_image_ids = get_project_image_ids(selected_metadata)

    # ----------------------------------------------------------------------------------------------- #
    # ------------------------------------------- Display ------------------------------------------- #
    # ----------------------------------------------------------------------------------------------- #

    st.subheader('Total Images')
    st.write(selected_metadata['total_images'])

    st.subheader('Total Experiments')
    st.write(selected_metadata['total_experiments'])

    st.subheader('Screens')
    if bool(selected_metadata['screens']): 
        for screen in selected_metadata['screens'].values():
            st.write(f'**Project Title**')
            st.write(f"_{screen['experiment_map_annotation']['Publication Title']}_")
            st.write(f'**Study Type**')
            st.write(screen['experiment_map_annotation']["Study Type"])
            st.write(f'**Total Datasets**')
            st.write(screen['total_datasets'])
            st.write(f'**Total Images in Project**')
            st.write(screen['total_images_in_project'])

    st.subheader('Projects')
    if bool(selected_metadata['projects']): 
        for project in selected_metadata['projects'].values():
            st.write(f'**Project Title**')
            st.write(f"_{project['experiment_map_annotation']['Publication Title']}_")
            st.write(f'**Study Type**')
            st.write(project['experiment_map_annotation']["Study Type"])
            st.write(f'**Total Datasets**')
            st.write(project['total_datasets'])
            st.write(f'**Total Images in Project**')
            st.write(project['total_images_in_project'])

    # Create button to resample images
    if st.button("Sample Images"):

        col1, col2 = st.columns(2)

        selected_screen_images = []
        selected_project_images = []

        if bool(selected_metadata['screens']):
            screen_image_ids = get_screen_image_ids(selected_metadata)
            # Randomly select images from "screens" and "projects"
            selected_screen_images = random.sample(screen_image_ids, 2)

        if bool(selected_metadata['projects']):
            project_image_ids = get_project_image_ids(selected_metadata)
            selected_project_images = random.sample(project_image_ids, 2)

        random_image_ids = selected_screen_images + selected_project_images

        # Display the images with the corresponding image_ids
        asyncio.run(display_images(random_image_ids))
