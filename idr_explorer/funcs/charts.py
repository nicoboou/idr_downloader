import streamlit as st
import plotly.graph_objects as go

def create_pie_chart(labels, values):
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    return fig

def create_bar_chart(data, hover_labels, categories, title):
    # Sort the data in descending order
    data, hover_labels, categories = zip(*sorted(zip(data, hover_labels, categories), reverse=True))

    # Get only last bit after_ in hover_labels
    hover_labels = [label.split('_')[-1] for label in hover_labels]

    colors = ['green' if category == 'cell' else 'blue' for category in categories]
    fig = go.Figure(data=go.Bar(x=hover_labels, y=data, hovertext=hover_labels, marker=dict(color=colors)))

    # Update the layout of the bar chart
    fig.update_layout(title=title)

    return fig

def create_microscopy_type_bar_chart(df, title):
    # Group the DataFrame by 'study_type' and calculate the sum of 'total_nb_images'
    grouped_df = df.groupby('study_type')['total_nb_images'].sum().reset_index()

    # Create the bar chart using graph_objects
    fig = go.Figure(
        data=go.Bar(
            x=grouped_df['study_type'],
            y=grouped_df['total_nb_images']
        )
    )

    # Update the layout of the bar chart
    fig.update_layout(
        title=title,
        xaxis_title='Study Type',
        yaxis_title='Number of Images'
    )

    return fig