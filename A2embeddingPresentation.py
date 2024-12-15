import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load datasets
celeba_s = pd.read_csv("datasets/celeba_buffalo_s_reworked.csv")
celeba_l = pd.read_csv("datasets/celeba_buffalo_l_reworked.csv")
s_embed = pd.read_csv("datasets/s_embed.csv")
l_embed = pd.read_csv("datasets/l_embed.csv")

# Define embedding and label columns
embedding_columns = ["embedding_" + str(i) for i in range(512)]
label_columns = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 
                 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 
                 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 
                 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 
                 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 
                 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

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
        name="Male"
    )

    trace_female = go.Scatter(
        x=embed_data["embed_" + selected_feature][label_data[selected_feature] == -1],
        y=embed_data["embed_not_" + selected_feature][label_data[selected_feature] == -1],
        mode='markers',
        marker=dict(color='blue', symbol='star'),
        name="Female"
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
app.layout = html.Div([
    html.H1("Embedding and Label Analysis", style={'textAlign': 'center'}),
    
    # Dropdowns for embedding and label selection
    html.Div([
        html.Div([
            html.Label("Select Embedding Column"),
            dcc.Dropdown(
                id='embedding-dropdown',
                options=[{'label': col, 'value': col} for col in embedding_columns],
                value=embedding_columns[0],  # Default value
                style={'width': '90%'}
            )
        ], style={'display': 'inline-block', 'width': '45%'}),
        
        html.Div([
            html.Label("Select Label Column"),
            dcc.Dropdown(
                id='label-dropdown',
                options=[{'label': col, 'value': col} for col in label_columns],
                value='Male',  # Default value
                style={'width': '90%'}
            )
        ], style={'display': 'inline-block', 'width': '45%'}),
    ], style={'padding': '20px'}),
    
    # Histograms
    html.Div([
        html.Div([
            dcc.Graph(id='celeba-s-histogram')
        ], style={'display': 'inline-block', 'width': '48%'}),
        
        html.Div([
            dcc.Graph(id='celeba-l-histogram')
        ], style={'display': 'inline-block', 'width': '48%'}),
    ]),
    
    # Scatter plots
    html.Div([
        html.Div([
            dcc.Graph(id='s-embedding-scatter')
        ], style={'display': 'inline-block', 'width': '48%'}),
        
        html.Div([
            dcc.Graph(id='l-embedding-scatter')
        ], style={'display': 'inline-block', 'width': '48%'}),
    ])
])

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
    app.run_server(debug=True)
