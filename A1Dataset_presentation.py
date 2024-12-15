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



fig1 = create_plotbar(percentages, "Labels", "Percentage (%)", "Percentage of Pictures with Each Label" )
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
    return create_plotbar(filtered_percentages, "Labels", "Percentage (%)", "Percentage of Pictures with Each Label" )




# App layout
label_popularity_block = html.Div([
    html.H3("Label Popularity", style={"marginBottom": "20px"}),
    dcc.Markdown(
    """
    The probability of an image having a label is not the same for all labels, some labels like "No_Beard" are very common with 85% of poeple pictures representing bald poeple, others like "Bald" are less popular with only 1.4% of pictures representing bald poeple.
    
    Of oparticular interest, we see that 4.8% of pictures are blurry thus perhaps difficultly exploitable.
    
    You can explore the labels with the interactive graphic hereunder:    
    """,
    style={'fontSize': 14}
    ),
    dcc.Dropdown(
        id="label-selector",
        options=[{"label": label, "value": label} for label in percentages.index],
        multi=True,
        placeholder="Select labels to display",
        style={"marginBottom": "20px"}
    ),
    dcc.Graph(id="label-bar-chart", figure=fig1),

], style=global_style)











consistency = df_s.groupby('id')[labels_columns].apply(lambda x: x.nunique() == 1)

# Calculate the percentage of IDs where the label is consistent
consistency_percentage = consistency.mean() * 100
consistency_percentage=consistency_percentage.sort_values(ascending=False)



fig2 = create_plotbar(consistency_percentage, "Labels", "Percentage (%)", "Percentage of poeple who have this label that vary for different pictures" )
# Callback to update the figure based on selected labels
@app.callback(
    dependencies.Output("label-consistency-bar-chart", "figure"),
    [dependencies.Input("label-consistency-selector", "value")]
)
def update_chart(selected_labels):
    if not selected_labels:
        filtered_consistency_percentage = consistency_percentage
    else:
        filtered_consistency_percentage = consistency_percentage[consistency_percentage.index.isin(selected_labels)]
    return create_plotbar(filtered_consistency_percentage, "Labels", "Percentage (%)", "Percentage of poeple who have this label that vary for different pictures" )






label_info_block = html.Div([
    html.H3("Poeple constantly labled the same and errors in the dataset", style={"marginBottom": "20px"}),
    dcc.Markdown(
    """
    When we have multiple pictures of the same person, for certain labels we expect the result to vary and for other not.

    For example, when studying label "Mouth_Slightly_Open", we expect that in certain pictures the person has this label = 1 and in other = 0.

    But for label "Male", the result must not vary for different pictures of the same person, however we see that for 11% of the persons have some male labeled pictures and some female labeled puictures.
    """,
    style={'fontSize': 14}
    ),
    dcc.Dropdown(
        id="label-consistency-selector",
        options=[{"label": label, "value": label} for label in consistency_percentage.index],
        multi=True,
        placeholder="Select labels to display",
        style={"marginBottom": "20px"}
    ),
    dcc.Graph(id="label-consistency-bar-chart", figure=fig2),
    dcc.Markdown(
        """    
        Do we have 11% trans poeple in the dataset?  No: this is due to a misslabeling of some pictures.
        """,
        style={'fontSize': 14} 
    ),
    html.Div(
    [
        html.Div(
            [
                html.Img(src=encode_image("img_celeba/163068.jpg"), style={"width": "45%"}),
                html.P("person 1631, labeled Male=-1", style={"textAlign": "center"}),
            ],
            style={"display": "inline-block", "textAlign": "center", "width": "50%"},
        ),
        html.Div(
            [
                html.Img(src=encode_image("img_celeba/172656.jpg"), style={"width": "45%"}),
                html.P("person 1631, labeled Male=1", style={"textAlign": "center"}),
            ],
            style={"display": "inline-block", "textAlign": "center", "width": "50%"},
        ),
    ],
    style={"display": "flex", "justifyContent": "space-around", "alignItems": "center"},
)

], style=global_style)






# App layout
dataset_presentation_page = html.Div([
    back_to_menu_block,
    header_block,
    label_exploration_block,
    label_popularity_block,
    label_info_block,
    back_to_menu_block,
])

# Run the app
if __name__ == "__main__":
    # Initialize Dash app
    app.layout=dataset_presentation_page
    app.run_server(debug=True)
