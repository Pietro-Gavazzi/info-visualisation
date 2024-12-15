from dash import Dash, dcc, html, dash_table, dependencies
import pandas as pd
import plotly.express as px
import pickle
from utils import *
import numpy as np

image_path = 'img_celeba/011256.jpg'  # Path to your image

human_columns = ["image_name", "Male", "Young", "Smiling", "Wearing_Hat", "Big_Nose"]



# Encode the image
encoded_image = encode_image(image_path)
# Load data and preprocess
with open('./datasets/inconsistent_ids_and_labels.pkl', 'rb') as file:
    inconsistent_ids_and_labels = pickle.load(file)

df_s, df_l = load_data()
labels_columns = list(inconsistent_ids_and_labels.keys())

# Calculate metrics
inconsistent_percentages = [(len(inconsistent_ids_and_labels[label]) / 1000) * 100 for label in labels_columns]




# Initialize Dash app
app = Dash(__name__)

# Define global style
global_style = {'marginLeft': '10%', 'marginRight': '10%'}

# Define content blocks
header_block = html.Div([
    html.H2("Human Labeling", style={"marginBottom": "20px"}),
    dcc.Markdown(
        """
        Before exploring both DNN's latent spaces, let's explore the human labelling of the pictures in the dataset.
        """,
        style={'fontSize': 14}
    ),
], style=global_style)


image_sub_block = html.Div([
    html.Figure([
        html.Img(src=encoded_image, style={'width': '50%'}),
        html.Figcaption("Image of a celebrity (011256.jpg)", style={'marginTop': '10px', 'fontSize': '14px', 'fontStyle': 'italic'})
    ], style={'textAlign': 'center', 'flex': 1, 'padding': '10px'}),

    html.Div([

        html.Div([
            dcc.Markdown(
                """
                    For example in the picture on the left, the person is "male", "young", "wearing a hat" but "not smilng" and "not with a big nose". 
                """,
                mathjax=True, style={'fontSize': 14}
            ),
        ]),

        dash_table.DataTable(
            id='table_human',
            columns=[{"name": col, "id": col} for col in human_columns],
            data=df_s[human_columns].head(1).to_dict('records'),
            style_table={'overflowX': 'auto', 'marginTop': '20px'},
            style_cell={'textAlign': 'center', 'fontSize': 14},
        ),

    ], style={'flex': 1, 'padding': '10px'}),

], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}
)


label_exploration_block = html.Div([
    html.H3("Basic overview", style={ "marginBottom": "20px"}),
    dcc.Markdown(
        """
        There are 39 different labels, manually attributed to each image by a human annotator.
        
        The labels in the dataset are represented by a flag which is equal to 1 if the label is present in the picture or set to -1 if the label is not present.   
        """,
        style={'fontSize': 14}
    ),
    image_sub_block,
], style=global_style)


percentages = (np.sum(df_s[labels_columns] == 1, axis=0) / len(df_s)) * 100
percentages = percentages.sort_values(ascending=False)

# Create the bar chart
fig1 = px.bar(
    x=percentages.index,
    y=percentages.values,
    labels={"x": "Labels", "y": "Percentage (%)"},
    title="Percentage of Pictures with Each Label",
)
fig1.update_layout(
    xaxis=dict(title="Labels", tickangle=45),
    yaxis=dict(title="Percentage (%)", tickvals=get_tickals(percentages)),
    margin=dict(l=40, r=40, t=40, b=120),
    height=600,
    plot_bgcolor="white",
    bargap=0.4,  # Adjust bar width to make them finer
    shapes=[
    dict(
        type="line",
        x0=-0.5,
        x1=len(percentages) - 0.5,
        y0=z,
        y1=z,
        line=dict(color="white", width=1)
    ) for z in get_tickals(percentages)
    ]
)
fig1.update_traces(
    marker_color="skyblue",
    marker_line_color="black",
    marker_line_width=1,
)


# Callback to update the figure based on selected labels
@app.callback(
    dependencies.Output("label-bar-chart", "figure"),
    [dependencies.Input("label-selector", "value")]
)
def update_chart(selected_labels):
    if not selected_labels:
        filtered_percentages = percentages
    else:
        filtered_percentages = percentages[percentages.index.isin(selected_labels)]

    # Create updated bar chart
    fig = px.bar(
        x=filtered_percentages.index,
        y=filtered_percentages.values,
        labels={"x": "Labels", "y": "Percentage (%)"},
    )
    fig.update_layout(
        xaxis=dict(title="Labels", tickvals=get_tickals(filtered_percentages)),
        yaxis=dict(title="Percentage (%)"),
        margin=dict(l=40, r=40, t=40, b=120),
        height=600,
        plot_bgcolor="white",
        bargap=0.4,  # Adjust bar width to make them finer
        shapes=[
        dict(
            type="line",
            x0=-0.5,
            x1=len(filtered_percentages) - 0.5,
            y0=z,
            y1=z,
            line=dict(color="white", width=1)
        ) for z in get_tickals(filtered_percentages)
        ]

    )
    fig.update_traces(
        marker_color="skyblue",
        marker_line_color="black",
        marker_line_width=1,
    )
    return fig



# App layout
label_popularity_block = html.Div([
    html.H3("Label Popularity", style={"marginBottom": "20px"}),
    dcc.Dropdown(
        id="label-selector",
        options=[{"label": label, "value": label} for label in percentages.index],
        multi=True,
        placeholder="Select labels to display",
        style={"marginBottom": "20px"}
    ),
    dcc.Graph(id="label-bar-chart", figure=fig1),

], style={"fontFamily": "Arial", "padding": "20px"})




# App layout
app.layout = html.Div([
    header_block,
    label_exploration_block,
    label_popularity_block
])

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
