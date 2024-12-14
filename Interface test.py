from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from utils import *

# Load and preprocess data
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

# Pre-compute t-SNE and PCA results
tsne_results_s = get_tsne_projection(buffalo_s_embed, perplexities=[30])[30]
tsne_results_l = get_tsne_projection(buffalo_l_embed, perplexities=[30])[30]
pca_results_s = get_pca_projection(buffalo_s_embed)
pca_results_l = get_pca_projection(buffalo_l_embed)

# Add IDs to dataframes
tsne_results_s['id'] = buffalo_s_label['id']
tsne_results_l['id'] = buffalo_l_label['id']
pca_results_s['id'] = buffalo_s_label['id']
pca_results_l['id'] = buffalo_l_label['id']


#store the figures in a dictionary

clustering_figures = {'kmeans': {}, 'dbscan': {}, 'dendrogram': {}}

# K-Means Clustering
kmeans_s = get_kmeans_clustering(tsne_results_s[['x', 'y']].values, n_clusters=3)
kmeans_l = get_kmeans_clustering(tsne_results_l[['x', 'y']].values, n_clusters=3)
kmeans_fig_s = px.scatter(kmeans_s, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_s)")
kmeans_fig_l = px.scatter(kmeans_l, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_l)")


# Generate dendrogram images for t-SNE and PCA
dendrogram_image_tsne_s = create_dendrogram_plot(tsne_results_s[['x', 'y']].values)
dendrogram_image_tsne_l = create_dendrogram_plot(tsne_results_l[['x', 'y']].values)
dendrogram_image_pca_s = create_dendrogram_plot(pca_results_s[['PCA1', 'PCA2']].values)
dendrogram_image_pca_l = create_dendrogram_plot(pca_results_l[['PCA1', 'PCA2']].values)

# Helper function to encode images in base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    

# Encode dendrogram images
#dendrogram_encoded_tsne_s = encode_image(dendrogram_image_tsne_s)
#dendrogram_encoded_tsne_l = encode_image(dendrogram_image_tsne_l)
#dendrogram_encoded_pca_s = encode_image(dendrogram_image_pca_s)
#dendrogram_encoded_pca_l = encode_image(dendrogram_image_pca_l)

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

    # t-SNE and PCA projections
    html.Div([
        html.H2("Projection Comparisons"),
        html.Div([
            html.Div([
                html.H3("Projection of buffalo_s"),
                dcc.Graph(id='projection-plot-s')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Projection of buffalo_l"),
                dcc.Graph(id='projection-plot-l')
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
                html.Img(id='dendrogram-plot-s')
            ], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([
                html.H3("Dendrogram for buffalo_l"),
                html.Img(id='dendrogram-plot-l')
            ], style={'display': 'inline-block', 'width': '49%'})
        ])
    ])
])

# Callback to update projection plots
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
    # Select appropriate data
    if projection_method == 'tsne':
        data_s, data_l = tsne_results_s.copy(), tsne_results_l.copy()
        x, y = 'x', 'y'
    elif projection_method == 'pca':
        data_s, data_l = pca_results_s.copy(), pca_results_l.copy()
        x, y = 'PCA1', 'PCA2'

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


if __name__ == '__main__':
    app.run_server(debug=True)
