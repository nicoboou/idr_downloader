import random
import os
import asyncio
import streamlit as st

# Local imports
from funcs.display import display_images, get_screen_image_ids, get_project_image_ids
from funcs.charts import create_pie_chart, create_bar_chart, create_microscopy_type_bar_chart
from funcs.dataframe import retrieve_dataframe

@st.cache_data()
def display_data_distribution(dataframe):

    # for screen in metadata_file['screens'].values():
    #     for plate in screen['plates'].values():
    #         for well in plate['wells'].values():
    #             for field in well['fields'].values():
    #                 total_nb_images += 1


    # Retrieve the necessary columns from the DataFrame
    nb_images_per_experiment = dataframe['total_nb_images'].tolist()
    categories = dataframe['study_type'].tolist()
    hover_labels = dataframe['metadata_file'].tolist()
    cell_experiments = dataframe[dataframe['sample_type'] == 'cell'].shape[0]
    tissue_experiments = dataframe[dataframe['sample_type'] == 'tissue'].shape[0]

    # Create and render the bar chart in Streamlit
    experiments_bar_chart = create_bar_chart(data=nb_images_per_experiment, hover_labels=hover_labels, categories=categories, title='Nb of Images per Experiment')
    st.plotly_chart(experiments_bar_chart)

    # Create and render the microscopy bar chart in Streamlit
    microscopy_bar_chart = create_microscopy_type_bar_chart(df=dataframe, title='Nb of Images per Microscopy Type')
    st.plotly_chart(microscopy_bar_chart)

    # Create & render a pie chart
    pie_chart = create_pie_chart(['cell', 'tissue'], [cell_experiments, tissue_experiments])
    st.plotly_chart(pie_chart)


    