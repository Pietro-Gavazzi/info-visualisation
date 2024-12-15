import pickle
from dash import Dash, html, dcc, Input, Output

# Load preprocessed data
with open("datasets/preprocessed2STEP_data.pkl", "rb") as f:
    data = pickle.load(f)

# Initialize Dash app
app = Dash(__name__)

# Layout with dropdown and dynamic graphs
app.layout = html.Div([
    html.H1("Clustering Multi-Step Results"),
    html.P(
        "Here we will visualize the results of the multi-step clustering process. "
        "Having a good embedding is great but it is also important to use the right clustering algorithm and hyperparameters to identify meaningful clusters. "
        "The data points are first projected into 39D space (corresponding to the number of labels to retain as much information as possible) using PCA, then clustered using K-Means, DBSCAN and Dendrograms. "
        "Finally, we use the labels computed on those 39-dimension data on the data projected into 2D space for visualization."
    ),

    html.Div([
        html.Label("Select Clustering Method:"),
        dcc.Dropdown(
            id='clustering-method',
            options=[
                {'label': 'K-Means', 'value': 'kmeans'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'Dendrogram', 'value': 'dendrogram'}
            ],
            value='kmeans',
            clearable=False
        ),
    ], style={'margin-bottom': '20px'}),

    html.Div(id='graphs-container')
])


# Callback to update graphs based on selected method
@app.callback(
    Output('graphs-container', 'children'),
    [Input('clustering-method', 'value')]
)
def update_graphs(selected_method):
    if selected_method == 'kmeans':
        return html.Div([
            html.H2("K-Means Clustering Results"),
            html.Div([
                html.Div([
                    html.H3("K-Means Step 1 (buffalo_s)"),
                    dcc.Graph(figure=data["kmeans_fig_s"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("K-Means Step 1 (buffalo_l)"),
                    dcc.Graph(figure=data["kmeans_fig_l"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.Div([
                    html.H3("K-Means Step 2 (buffalo_s)"),
                    dcc.Graph(figure=data["kmeans_fig_s2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("K-Means Step 2 (buffalo_l)"),
                    dcc.Graph(figure=data["kmeans_fig_l2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ])
        ])

    elif selected_method == 'dbscan':
        return html.Div([
            html.H2("DBSCAN Clustering Results"),
            html.Div([
                html.Div([
                    html.H3("DBSCAN Step 1 (buffalo_s)"),
                    dcc.Graph(figure=data["dbscan_fig_s"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("DBSCAN Step 1 (buffalo_l)"),
                    dcc.Graph(figure=data["dbscan_fig_l"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.Div([
                    html.H3("DBSCAN Step 2 (buffalo_s)"),
                    dcc.Graph(figure=data["dbscan_fig_s2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("DBSCAN Step 2 (buffalo_l)"),
                    dcc.Graph(figure=data["dbscan_fig_l2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ])
        ])

    elif selected_method == 'dendrogram':
        return html.Div([
            html.H2("Dendrogram Clustering Results"),
            html.Div([
                html.Div([
                    html.H3("Dendrogram in 2D (buffalo_s)"),
                    dcc.Graph(figure=data["dendrogram_fig_s2D"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Dendrogram in 2D (buffalo_l)"),
                    dcc.Graph(figure=data["dendrogram_fig_l2D"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            html.Div([
                html.Div([
                    html.H3("Dendrogram in Multi-Step Clustering (buffalo_s)"),
                    dcc.Graph(figure=data["dendrogram_fig_s2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3("Dendrogram in Multi-Step Clustering (buffalo_l)"),
                    dcc.Graph(figure=data["dendrogram_fig_l2STEP"])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ])
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
