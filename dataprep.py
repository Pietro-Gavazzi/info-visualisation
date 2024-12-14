import os
import pickle
from utils import *
import pandas as pd

# Ensure the datasets directory exists
os.makedirs("./datasets", exist_ok=True)

# Load and preprocess data
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

# Pre-compute t-SNE and PCA results
tsne_results_s = get_tsne_projection(buffalo_s_embed, perplexities=[30])[30]
tsne_results_l = get_tsne_projection(buffalo_l_embed, perplexities=[30])[30]
#pca_results_s = get_pca_projection(buffalo_s_embed)
#pca_results_l = get_pca_projection(buffalo_l_embed)

# Add IDs to dataframes
tsne_results_s['id'] = buffalo_s_label['id']
tsne_results_l['id'] = buffalo_l_label['id']
#pca_results_s['id'] = buffalo_s_label['id']
#pca_results_l['id'] = buffalo_l_label['id']

# K-Means Clustering
kmeans_s = get_kmeans_clustering(tsne_results_s[['x', 'y']].values, n_clusters=3)
kmeans_l = get_kmeans_clustering(tsne_results_l[['x', 'y']].values, n_clusters=3)
kmeans_fig_s = px.scatter(kmeans_s, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_s)")
kmeans_fig_l = px.scatter(kmeans_l, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_l)")

# DBSCAN Clustering
dbscan_s = get_dbscan_clustering(tsne_results_s[['x', 'y']].values, eps=3, min_samples=5)
dbscan_l = get_dbscan_clustering(tsne_results_l[['x', 'y']].values, eps=3, min_samples=5)
dbscan_fig_s = px.scatter(dbscan_s, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_s)")
dbscan_fig_l = px.scatter(dbscan_l, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_l)")

# Generate dendrogram images for t-SNE and PCA
dendrogram_image_tsne_s = create_dendrogram_plot(tsne_results_s[['x', 'y']].values)
dendrogram_image_tsne_l = create_dendrogram_plot(tsne_results_l[['x', 'y']].values)
#dendrogram_image_pca_s = create_dendrogram_plot(pca_results_s[['PCA1', 'PCA2']].values)
#dendrogram_image_pca_l = create_dendrogram_plot(pca_results_l[['PCA1', 'PCA2']].values)

# Save everything to /datasets
data_to_save = {
    "buffalo_s": buffalo_s,
    "buffalo_l": buffalo_l,
    "tsne_results_s": tsne_results_s,
    "tsne_results_l": tsne_results_l,
    #"pca_results_s": pca_results_s,
    #"pca_results_l": pca_results_l,
    "kmeans_fig_s": kmeans_fig_s,
    "kmeans_fig_l": kmeans_fig_l,
    "dbscan_fig_s": dbscan_fig_s,
    "dbscan_fig_l": dbscan_fig_l,
    "dendrogram_image_tsne_s": dendrogram_image_tsne_s,
    "dendrogram_image_tsne_l": dendrogram_image_tsne_l,
    #"dendrogram_image_pca_s": dendrogram_image_pca_s,
    #"dendrogram_image_pca_l": dendrogram_image_pca_l,
}

with open("./datasets/preprocessed_data.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print("Data preparation complete. All datasets saved to /datasets.")
