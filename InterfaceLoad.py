import pickle
from dash import Dash, html, dcc, Input, Output
import plotly.express as px


# Load precomputed data
with open("datasets/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Unpack data
buffalo_s = data["buffalo_s"]
buffalo_l = data["buffalo_l"]
tsne_results_s = data["tsne_results_s"]
tsne_results_l = data["tsne_results_l"]
#pca_results_s = data["pca_results_s"]
#pca_results_l = data["pca_results_l"]
kmeans_fig_s = data["kmeans_fig_s"]
kmeans_fig_l = data["kmeans_fig_l"]

dbscan_fig_s = data["dbscan_fig_s"]
dbscan_fig_l = data["dbscan_fig_l"]

dendrogram_image_tsne_s = data["dendrogram_image_tsne_s"]
dendrogram_image_tsne_l = data["dendrogram_image_tsne_l"]
#dendrogram_image_pca_s = data["dendrogram_image_pca_s"]
#dendrogram_image_pca_l = data["dendrogram_image_pca_l"]

# Initialize Dash app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Buffalo Dataset Analysis"),

    # Dropdown to select an ID for highlighting
    html.Div([
        html.Label("Select ID to Highlight:"),
        dcc.Dropdown(
            options=[{'label': f'ID {i}', 'value': i} for i in tsne_results_s['id'].unique()],
            id='id-selector',
            value=None,
            clearable=True
        )
    ]),

    # Dropdown to select projection method for clustering and dendrograms
    html.Div([
        html.Label("Select Projection Method for Clustering:"),
        dcc.Dropdown(
            options=[
                {'label': 't-SNE', 'value': 'tsne'},
                {'label': 'PCA', 'value': 'pca'}
            ],
            id='clustering-method',
            value='tsne',
            clearable=False
        )
    ]),

    # Projections and visualizations
    html.Div([
        html.Div([
            html.H3("Projection of buffalo_s"),
            dcc.Graph(id='projection-plot-s')
        ], style={'display': 'inline-block', 'width': '49%'}),
        html.Div([
            html.H3("Projection of buffalo_l"),
            dcc.Graph(id='projection-plot-l')
        ], style={'display': 'inline-block', 'width': '49%'})
    ]),

    html.Div([
        html.H2("Dendrogram Clustering"),
        html.Div([
            html.Div([
                html.H3("Dendrogram for buffalo_s"),
                html.Img(id='dendrogram-plot-s')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Dendrogram for buffalo_l"),
                html.Img(id='dendrogram-plot-l')
            ], style={'display': 'inline-block', 'width': '49%'})
        ])
    ]),
    # K-Means Clustering
    html.Div([
        html.H2("K-Means Clustering"),
        html.Div([
            html.Div([
                html.H3("Clustering on buffalo_s"),
                dcc.Graph(id='kmeans-plot-s')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Clustering on buffalo_l"),
                dcc.Graph(id='kmeans-plot-l')
            ], style={'display': 'inline-block', 'width': '49%'})
        ])
    ]),

    # DBSCAN Clustering
    html.Div([
        html.H2("DBSCAN Clustering"),
        html.Div([
            html.Div([
                html.H3("Clustering on buffalo_s"),
                dcc.Graph(id='dbscan-plot-s')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Clustering on buffalo_l"),
                dcc.Graph(id='dbscan-plot-l')
            ], style={'display': 'inline-block', 'width': '49%'})
        ])
    ]),

    # Dendrogram
    html.Div([
        html.H2("Dendrogram Clustering"),
        html.Div([
            html.Div([
                html.H3("Dendrogram for buffalo_s"),
                html.Img(id='dendrogram-plot-s',width= '99%')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Dendrogram for buffalo_l"),
                html.Img(id='dendrogram-plot-l',width= '99%')
            ], style={'display': 'inline-block', 'width': '49%'})
        ])
    ])
])

# Callbacks for projections
@app.callback(
    [Output('projection-plot-s', 'figure'),
     Output('projection-plot-l', 'figure')],
    Input('id-selector', 'value')
)
def update_projection_plots(selected_id):
    # Highlight selected ID
    tsne_s = tsne_results_s.copy()
    tsne_l = tsne_results_l.copy()
    tsne_s['color'] = tsne_s['id'].apply(lambda i: 'red' if str(i) == str(selected_id) else 'blue')
    tsne_l['color'] = tsne_l['id'].apply(lambda i: 'red' if str(i) == str(selected_id) else 'blue')

    # Generate projection figures
    proj_fig_s = px.scatter(
    tsne_s, x='x', y='y', color='color',
    title="Projection of buffalo_s",
    color_discrete_map={'red': 'red', 'blue': 'blue'}
    )
    proj_fig_l = px.scatter(
    tsne_l, x='x', y='y', color='color',
    title="Projection of buffalo_l",
    color_discrete_map={'red': 'red', 'blue': 'blue'}
    )

    return proj_fig_s, proj_fig_l

# Callback to update clustering and dendrograms
@app.callback(
    [Output('kmeans-plot-s', 'figure'),
     Output('kmeans-plot-l', 'figure'),
     Output('dbscan-plot-s', 'figure'),
     Output('dbscan-plot-l', 'figure'),
     Output('dendrogram-plot-s', 'src'),
     Output('dendrogram-plot-l', 'src')],
    Input('clustering-method', 'value')
)
def update_clustering_plots(projection_method):
    
    
    with open("datasets/preprocessed_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Unpack data
    buffalo_s = data["buffalo_s"]
    buffalo_l = data["buffalo_l"]
    tsne_results_s = data["tsne_results_s"]
    tsne_results_l = data["tsne_results_l"]
    #pca_results_s = data["pca_results_s"]
    #pca_results_l = data["pca_results_l"]
    kmeans_fig_s = data["kmeans_fig_s"]
    kmeans_fig_l = data["kmeans_fig_l"]

    dbscan_fig_s = data["dbscan_fig_s"]
    dbscan_fig_l = data["dbscan_fig_l"]

    dendrogram_image_tsne_s = data["dendrogram_image_tsne_s"]
    dendrogram_image_tsne_l = data["dendrogram_image_tsne_l"]
    #dendrogram_image_pca_s = data["dendrogram_image_pca_s"]
    #dendrogram_image_pca_l = data["dendrogram_image_pca_l"]
    # Select appropriate data
    if projection_method == 'tsne':
        return (
            kmeans_fig_s,
            kmeans_fig_l,
            dbscan_fig_s,
            dbscan_fig_l,
            f"data:image/png;base64,{dendrogram_image_tsne_s}",
            f"data:image/png;base64,{dendrogram_image_tsne_l}"
        )
        
    elif projection_method == 'pca':
        #data_s, data_l = pca_results_s.copy(), pca_results_l.copy()
        x, y = 'PCA1', 'PCA2'
        return (
            kmeans_fig_s,
            kmeans_fig_l,
            dbscan_fig_s,
            dbscan_fig_l,
            f"data:image/png;base64,{dendrogram_image_tsne_s}",
            f"data:image/png;base64,{dendrogram_image_tsne_l}"
        )

    # K-Means Clustering
    kmeans_s = get_kmeans_clustering(data_s[[x, y]].values, n_clusters=3)
    kmeans_l = get_kmeans_clustering(data_l[[x, y]].values, n_clusters=3)
    kmeans_fig_s = px.scatter(kmeans_s, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_s)")
    kmeans_fig_l = px.scatter(kmeans_l, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_l)")

    # DBSCAN Clustering
    dbscan_s = get_dbscan_clustering(data_s[[x, y]].values, eps=3, min_samples=5)
    dbscan_l = get_dbscan_clustering(data_l[[x, y]].values, eps=3, min_samples=5)
    dbscan_fig_s = px.scatter(dbscan_s, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_s)")
    dbscan_fig_l = px.scatter(dbscan_l, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_l)")

    # Dendrograms
    if projection_method == 'tsne':
         dendrogram_s, dendrogram_l = f"data:image/png;base64,{dendrogram_image_tsne_s}", f"data:image/png;base64,{dendrogram_image_tsne_l}"
    elif projection_method == 'pca':
         dendrogram_s, dendrogram_l = f"data:image/png;base64,{dendrogram_image_pca_s}", f"data:image/png;base64,{dendrogram_image_pca_l}"
    

    return kmeans_fig_s, kmeans_fig_l, dbscan_fig_s, dbscan_fig_l, dendrogram_s, dendrogram_l

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
