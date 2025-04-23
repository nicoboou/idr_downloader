import math
import streamlit as st
import plotly.graph_objects as go

def display_shannon_entropy(dataframe):
    """
    Display the Shannon's Entropy of the selected rows in the dataframe in a gauge plot.

    Parameters
    ----------
    dataframe (pandas.DataFrame): the dataframe containing the selected rows.
    
    Returns
    -------
    None
    """
    
    category_counts = dataframe['study_type'].value_counts()
    total_count = len(dataframe)

    probabilities = category_counts / total_count

    entropy = -sum(probabilities * probabilities.apply(math.log2))

    fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=entropy,
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 2]},  # Customize the range of the gauge axis
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 1,
            'value': entropy
        }
    },
    title={'text': "Shannon's Entropy"}
    ))

    st.plotly_chart(fig, use_container_width=True)
