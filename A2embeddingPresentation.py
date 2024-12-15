import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from utils import *

# Load datasets
celeba_s = pd.read_csv("datasets/celeba_buffalo_s_reworked.csv")
celeba_l = pd.read_csv("datasets/celeba_buffalo_l_reworked.csv")
s_embed = pd.read_csv("datasets/s_embed.csv")
l_embed = pd.read_csv("datasets/l_embed.csv")

global_style = {'marginLeft': '10%', 'marginRight': '10%'}

# Function to create histograms
def create_histogram(data, column, title):
    return go.Figure(
        data=[
            go.Histogram(
                x=data[column],
                nbinsx=30,
                marker=dict(color='purple', line=dict(width=1, color='black')),
                opacity=0.7
            )
        ],
        layout=go.Layout(
            title=title,
            xaxis=dict(title=column),
            yaxis=dict(title="Frequency"),
            bargap=0.2
        )
    )

# Function to create scatter plots
def create_scatter(embed_data, label_data, selected_feature):
    trace_male = go.Scatter(
        x=embed_data["embed_" + selected_feature][label_data[selected_feature] == 1],
        y=embed_data["embed_not_" + selected_feature][label_data[selected_feature] == 1],
        mode='markers',
        marker=dict(color='red', symbol='star'),
        name=selected_feature
    )

    trace_female = go.Scatter(
        x=embed_data["embed_" + selected_feature][label_data[selected_feature] == -1],
        y=embed_data["embed_not_" + selected_feature][label_data[selected_feature] == -1],
        mode='markers',
        marker=dict(color='blue', symbol='star'),
        name="not_"+selected_feature
    )

    layout = go.Layout(
        title=f"Scatter Plot of Embeddings by {selected_feature}",
        xaxis=dict(title=f"Embed {selected_feature}"),
        yaxis=dict(title=f"Embed Not {selected_feature}"),
        showlegend=True
    )

    return go.Figure(data=[trace_male, trace_female], layout=layout)

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
embedding_presentation =html.Div([
    back_to_menu_block,
    html.H2("DNN Labbeling", style={'textAlign': 'center'}),
    html.H3("A little histogram", style=global_style),
    dcc.Markdown(
    """
    Embeddings are difficult to interpret with the naked eye, if we just see plot an histogram of all values for all 511 embeddings we will only see a boring and uninterpretable gaussian behavior.   
    """,
    style=global_style
    ),
    # Dropdowns for embedding and label selection
    html.Div([
        html.Label("Select Embedding Column"),
        dcc.Dropdown(
            id='embedding-dropdown',
            options=[{'label': col, 'value': col} for col in embedding_columns],
            value=embedding_columns[0],  # Default value
            style={'width': '90%'}
        )
    ], style={'display': 'inline-block', 'width': '45%'}),
        


    # Histograms
    html.Div([
        html.Div([
            dcc.Graph(id='celeba-s-histogram')
        ], style={'display': 'inline-block', 'width': '48%'}),
        
        html.Div([
            dcc.Graph(id='celeba-l-histogram')
        ], style={'display': 'inline-block', 'width': '48%'}),
    ], style = global_style),



    html.H3("The secrets of the embedding space", style=global_style),
    dcc.Markdown(
    """
    What was discovered in neural network representation is that if we take the mean of the embeddings of the pictures with a specific label, this vector of mean embedding can be interpreted as the the dnn representation of the mabel.

    This has inspire our linear algorithm for dimensionality reduction: we chose some features we want to study, and the scalar product between the normalised mean vector of the label (mean of all pictures label=1) and a picture, and the mean vector of the not_label (mean of all pictures label=-1) and a picture are two new features called respectively "embed label" and "embed not label".  

    You can see here under a plot of the data in these two new features for the chosen label.       
    """,
    style=global_style
    ),



    html.Div([
        html.Label("Select Label Column"),
        dcc.Dropdown(
            id='label-dropdown',
            options=[{'label': col, 'value': col} for col in labels_columns],
            value='Male',  # Default value
            style={'width': '90%'}
        )
    ], style={'display': 'inline-block', 'width': '45%'}),


    # Scatter plots
    html.Div([
        html.Div([
            dcc.Graph(id='s-embedding-scatter')
        ], style={'display': 'inline-block', 'width': '48%'}),
        
        html.Div([
            dcc.Graph(id='l-embedding-scatter')
        ], style={'display': 'inline-block', 'width': '48%'}),
    ], style = global_style),
    back_to_menu_block
], style = global_style)



def register_callbacksA2Page(app):
    # Callbacks to update histograms and scatter plots
    @app.callback(
        [Output('celeba-s-histogram', 'figure'),
        Output('celeba-l-histogram', 'figure'),
        Output('s-embedding-scatter', 'figure'),
        Output('l-embedding-scatter', 'figure')],
        [Input('embedding-dropdown', 'value'),
        Input('label-dropdown', 'value')]
    )
    def update_plots(selected_embedding, selected_label):
        # Histograms
        s_hist = create_histogram(celeba_s, selected_embedding, "Celeba S Histogram")
        l_hist = create_histogram(celeba_l, selected_embedding, "Celeba L Histogram")
        
        # Scatter plots
        s_scatter = create_scatter(s_embed, celeba_s, selected_label)
        l_scatter = create_scatter(l_embed, celeba_l, selected_label)
        
        return s_hist, l_hist, s_scatter, l_scatter

# Run the app
if __name__ == '__main__':
    register_callbacksA2Page(app)
    app.layout = embedding_presentation
    app.run_server(debug=True)
