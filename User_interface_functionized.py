import pandas as pd
import numpy as np
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
import umap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from utils import *


# Example usage within Dash application
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

scatter_df = get_projection(buffalo_l_embed, buffalo_l_label)
label_occurrences = get_label_occurrences(buffalo_l_label)
tsne_results = get_tsne_projection(buffalo_l_embed, perplexities=[30])
pca_results = get_pca_projection(buffalo_l_embed)
umap_results = get_umap_projection(buffalo_l_embed)
kmeans_results = get_kmeans_clustering(tsne_results[30].values, n_clusters=3)
#hierarchical_fig = get_hierarchical_clustering(buffalo_l_embed)
dendrogram_image = create_dendrogram_plot(buffalo_l_embed)
dbscan_results = get_dbscan_clustering(tsne_results[30].values, eps=3, min_samples=5)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Buffalo Dataset Analysis"),

    html.H2("Projection of Blurry vs Non-Blurry"),
    dcc.Graph(
        figure=px.scatter(
            scatter_df, x='x', y='y', color='Category',
            title="Blurry vs Non-Blurry Projections",
            labels={'x': 'Projection X', 'y': 'Projection Y'}
        )
    ),

    html.H2("Occurrences of Each Label in Dataset"),
    dcc.Graph(
        figure=px.bar(
            x=label_occurrences.index, y=label_occurrences.values,
            title="Occurrences of Each Label in Dataset",
            labels={'x': 'Label', 'y': 'Number of Occurrences'}
        )
    ),

    html.H2("t-SNE Projection (Perplexity: 30)"),
    dcc.Graph(
        figure=px.scatter(
            tsne_results[30], x='x', y='y',
            title="t-SNE Projection (Perplexity: 30)",
            labels={'x': 't-SNE X', 'y': 't-SNE Y'}
        )
    ),

    html.H2("PCA Projection"),
    dcc.Graph(
        figure=px.scatter(
            pca_results, x='PCA1', y='PCA2',
            title="PCA Projection of Embeddings",
            labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'}
        )
    ),

    html.H2("UMAP Projection"),
    dcc.Graph(
        figure=px.scatter(
            umap_results, x='UMAP1', y='UMAP2',
            title="UMAP Projection of Embeddings",
            labels={'UMAP1': 'UMAP Dimension 1', 'UMAP2': 'UMAP Dimension 2'}
        )
    ),
    html.H2("K-Means Clustering"),
    dcc.Graph(
        figure=px.scatter(
            kmeans_results, x='x', y='y', color='Cluster',
            title="K-Means Clustering Visualization",
            labels={'x': 't-SNE X', 'y': 't-SNE Y', 'color': 'Cluster'}
        )
    ),

    # Hierarchical clustering dendrogram
    html.H2("Hierarchical Clustering Dendrogram"),
    html.Img(src=f"data:image/png;base64,{dendrogram_image}")
    ,
    

    html.H2("DBSCAN Clustering"),
    dcc.Graph(
        figure=px.scatter(
            dbscan_results, x='x', y='y', color='Cluster',
            title="DBSCAN Clustering Visualization",
            labels={'x': 'Cluster X', 'y': 'Cluster Y', 'color': 'Cluster'}
        )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
    print("Done")