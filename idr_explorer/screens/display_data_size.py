import random
import os
import asyncio
import streamlit as st

# Local imports
from funcs.display import display_images, get_screen_image_ids, get_project_image_ids
from funcs.charts import create_pie_chart

def display_data_size():
    # Example data
    labels = ['Label 1', 'Label 2', 'Label 3']
    values = [30, 40, 60]

    # Create a pie chart
    pie_chart = create_pie_chart(labels, values)

    # Render the pie chart in Streamlit
    st.plotly_chart(pie_chart)