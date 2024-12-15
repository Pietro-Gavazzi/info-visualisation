# CPage.py
import pickle
from dash import html, dcc, Input, Output

# Load preprocessed data
with open("datasets/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Define the page layout
Cpage = html.Div([
    html.Div([
        html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
    ], style={"justify-content": "center", 'alignItems': 'center'}),
    html.H1("Clustering Multi-Step Results"),
    html.P(
        "Here we will visualize the results of the multi-step clustering process. "
        "Having a good embedding is great but it is also important to use the right clustering algorithm and hyperparameters to identify meaningful clusters. "
        "The data points are first projected into 39D space (corresponding to the number of labels to retain as much information as possible) using PCA, then clustered using K-Means, a density based algorithm : DBSCAN and a hierarchical algorithm : Linkage. "
        "Finally, we use the labels computed on those 39-dimension data on the data projected into 2D space for visualization."
    ),

    html.Div([
        html.Label("Select Clustering Method:"),
        dcc.Dropdown(
            id='clustering-method',
            options=[
                {'label': 'K-Means', 'value': 'kmeans'},
                {'label': 'DBSCAN', 'value': 'dbscan'},
                {'label': 'Linkage', 'value': 'linkage'}
            ],
            value='kmeans',
            clearable=False
        ),
    ]),
    html.Div(id='graphs-container'),
    
    html.H1("Clustering Multi-Step Results"),
    # Static text explanation
    html.Div([
        html.H2("Primary Analysis"),
        html.P(
            "We can observe that all the clustering algorithms have successfully identified clusters in the data. "
            "In all cases, K-means and Linkage seem to have created more distinct clusters and shows similar patterns while DBSCAN shows a big cluster in the center and more misclassified points. "
            "DBSCAN is also more sensitive to the choice of hyperparameters, such as the epsilon and min_samples values. "
            "This can explain the presence of misslabelled points in the DBSCAN clusters. "
            "We can also see that the clusters are more clearly separated using the 39D space labelling than the 2D space labelling. "
            "Especially, the center tends to create larger clusters in 2D space. "
            "This demonstrates the importance of clustering in a higher-dimensional space before projecting the data into 2D space for visualization. "
            "We can also observe that the clusters are more well separated in the buffalo_l dataset than in the buffalo_s dataset, indicating that the buffalo_l dataset distincts more the clusters than the buffalo_s dataset."
            "Especially at the center of the datasets where the points are closer to each other."
            "This also come with the drawback that the remaining center points are more difficult to cluster since they seem to have been labeled 'randomly',"
            "but this is better than having bigger wrongly labeled clusters at the center." 
        ),
    ], style={'margin-bottom': '20px', 'padding': '10px', 'backgroundColor': '#f9f9f9', 'border': '1px solid #ddd'}),
    
    html.Div([
        html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
    ], style={"justify-content": "center", 'alignItems': 'center'}),
])

# Define callbacks (these should be registered within the main app)
def register_callbacksCPage(app):
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
        elif selected_method == 'linkage':
            return html.Div([
                html.H2("Linkage Clustering Results"),
                html.Div([
                    html.Div([
                        html.H3("Linkage in 2D (buffalo_s)"),
                        dcc.Graph(figure=data["dendrogram_fig_s2D"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("Linkage in 2D (buffalo_l)"),
                        dcc.Graph(figure=data["dendrogram_fig_l2D"])
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ]),
                html.Div([
                    html.Div([
                        html.H3("Linkage in Multi-Step Clustering (buffalo_s)"),
                        dcc.Graph(figure=data["dendrogram_fig_s2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    html.Div([
                        html.H3("Linkage in Multi-Step Clustering (buffalo_l)"),
                        dcc.Graph(figure=data["dendrogram_fig_l2STEP"])
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ])
            ])
