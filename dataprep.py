import os
import pickle
from utils import *
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import plotly.express as px



# Ensure the datasets directory exists
os.makedirs("./datasets", exist_ok=True)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def filterData():
    buffalo_s = pd.read_csv("datasets/celeba_buffalo_s.csv")
    buffalo_l = pd.read_csv("datasets/celeba_buffalo_l.csv")

    # Find common image names
    common_image_names = set(buffalo_s['image_name']).intersection(set(buffalo_l['image_name']))

    # Filter rows where image_name is in the common set
    buffalo_s_filtered = buffalo_s[buffalo_s['image_name'].isin(common_image_names)]
    buffalo_l_filtered = buffalo_l[buffalo_l['image_name'].isin(common_image_names)]

    # order the data by image_name
    buffalo_s_filtered = buffalo_s_filtered.sort_values(by=['id', 'image_name'])
    buffalo_l_filtered = buffalo_l_filtered.sort_values(by=['id', 'image_name'])

    # Reorder columns to make 'id' the first column
    buffalo_s_filtered = buffalo_s_filtered[['id'] + [col for col in buffalo_s_filtered.columns if col != 'id']]
    buffalo_l_filtered = buffalo_l_filtered[['id'] + [col for col in buffalo_l_filtered.columns if col != 'id']]



    #save the filtered data
    buffalo_s_filtered.to_csv("datasets/celeba_buffalo_s_reworked.csv", index=False)
    buffalo_l_filtered.to_csv("datasets/celeba_buffalo_l_reworked.csv", index=False)


    # Verify
    print("Number of common rows:", len(common_image_names))
    print ("Number of rows in buffalo_s_filtered:", len(buffalo_s_filtered))
    print ("Number of rows in buffalo_l_filtered:", len(buffalo_l_filtered))
filterData()


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
features_label = labels_columns

df_s, df_l = load_data()
embed_s = np.array(df_s[embedding_columns])
embed_l = np.array(df_l[embedding_columns])

def projection(V, dataset):
    return V@dataset.T/np.linalg.norm(V)


new_s = df_s[id_columns+image_name_columns].copy()
new_l = df_l[id_columns+image_name_columns].copy()

new_columns = []
for label in features_label:
    print(label)
    
    indices_label = df_l[label]==1
    indices_not_label = df_l[label]==-1

    l_v_label = np.array(np.sum(df_l[embedding_columns][indices_label]))
    new_l["embed_"+label] = projection(l_v_label, embed_l)


    l_v_not_label = np.sum(df_l[embedding_columns][indices_not_label]) 
    new_l["embed_not_"+label] = projection(l_v_not_label, embed_l)
  

    s_v_label = np.sum(df_s[embedding_columns][indices_label])
    new_s["embed_"+label] = projection(s_v_label, embed_s)

    s_v_not_label = np.sum(df_s[embedding_columns][indices_not_label])
    new_s["embed_not_"+label] = projection(s_v_not_label, embed_s)

    new_columns.append("embed_"+label)
    new_columns.append("embed_not_"+label)


new_l.to_csv("datasets/l_embed.csv")
new_s.to_csv("datasets/s_embed.csv")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Load and preprocess data
buffalo_s, buffalo_l = load_data()
buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label = preprocess_data(buffalo_s, buffalo_l)

print("Beginning data preparation TSNE...")  
# # Pre-compute t-SNE and PCA results
# tsne_results_s3 = get_tsne_projection(buffalo_s_embed, perplexities=[3])[3]
# tsne_results_l3 = get_tsne_projection(buffalo_l_embed, perplexities=[3])[3]
# print("TSNE 3 complete.")
tsne_results_s30 = get_tsne_projection(buffalo_s_embed, perplexities=[30])[30]
tsne_results_l30 = get_tsne_projection(buffalo_l_embed, perplexities=[30])[30]

print("TSNE 30 complete.")

# print ("Beginning data preparation 2STEP...")
tsne_39D_l = get_pca_projection(buffalo_l_embed, n_components=39)
tsne_39D_s = get_pca_projection(buffalo_s_embed, n_components=39)
#tsne_39D_l, tsne_39D_s = get_embed_projection()

tsne_39D_l["id"]=buffalo_l_label["id"]
tsne_39D_s["id"]=buffalo_s_label["id"]

# # K-Means Clustering
print("2STEP data preparation complete.")



print("Beginning K-Means clustering on 39D t-SNE results...")

# # Step 2: K-Means clustering on the 39D t-SNE results
kmeans_clusters_s = get_kmeans_clustering(tsne_39D_s.drop(columns='id').values, n_clusters=1000)
kmeans_clusters_l = get_kmeans_clustering(tsne_39D_l.drop(columns='id').values, n_clusters=1000)

# # Add cluster labels to dataframes
tsne_39D_s['Cluster'] = kmeans_clusters_s['Cluster']
tsne_39D_l['Cluster'] = kmeans_clusters_l['Cluster']

print("K-Means clustering complete.")

print("Beginning t-SNE projection into 2D space for visualization...")

tsne_2D_s = tsne_results_s30
tsne_2D_l = tsne_results_l30



# # Step 3: Second t-SNE projection into 2D space for visualization
tsne_2D_s = get_tsne_projection(tsne_39D_s.drop(columns=['id', 'Cluster']).values, perplexities=[30], n_components=2)[30]
tsne_2D_l = get_tsne_projection(tsne_39D_l.drop(columns=['id', 'Cluster']).values, perplexities=[30], n_components=2)[30]

# # Convert t-SNE 2D results to dataframes and add IDs and cluster labels
tsne_2D_s = pd.DataFrame(tsne_2D_s, columns=['x', 'y'])
tsne_2D_s['id'] = tsne_39D_s['id']
tsne_2D_s['Cluster'] = tsne_39D_s['Cluster']

tsne_2D_l = pd.DataFrame(tsne_2D_l, columns=['x', 'y'])
tsne_2D_l['id'] = tsne_39D_l['id']
tsne_2D_l['Cluster'] = tsne_39D_l['Cluster']



# # Visualization of K-Means clusters on 2D t-SNE results
kmeans_fig_s2STEP = px.scatter(tsne_2D_s, x='x', y='y', color=tsne_39D_s["Cluster"], title="K-Means Clustering (Buffalo S - 2D t-SNE)")
kmeans_fig_l2STEP = px.scatter(tsne_2D_l, x='x', y='y', color=tsne_39D_l["Cluster"], title="K-Means Clustering (Buffalo L - 2D t-SNE)")

print("t-SNE 2D projection complete.")

print("Beginning DBSCAN clustering on 39D t-SNE results...")
# # Optional: DBSCAN Clustering (if needed)
dbscan_clusters_s = get_dbscan_clustering(tsne_39D_s.drop(columns='id').values, eps=10, min_samples=4)
dbscan_clusters_l = get_dbscan_clustering(tsne_39D_l.drop(columns='id').values, eps=10, min_samples=4)

# # Add DBSCAN cluster labels to 39D t-SNE data
#print('Cluster' in dbscan_clusters_s.columns)
tsne_39D_s['DBSCAN_Cluster'] = dbscan_clusters_s['Cluster']
tsne_39D_l['DBSCAN_Cluster'] = dbscan_clusters_l['Cluster']



# # Visualization of DBSCAN clusters on 2D t-SNE results
dbscan_fig_s2STEP = px.scatter(tsne_2D_s, x='x', y='y', color=tsne_39D_s['DBSCAN_Cluster'], title="DBSCAN Clustering (Buffalo S - 2D t-SNE)")
dbscan_fig_l2STEP = px.scatter(tsne_2D_l, x='x', y='y', color=tsne_39D_l['DBSCAN_Cluster'], title="DBSCAN Clustering (Buffalo L - 2D t-SNE)")

print("DBSCAN clustering complete.")


print ("Beginning data preparation Linkage...")
linkage_s = linkage(tsne_39D_s.drop(columns=['id']).values, method='ward')  # Ward's method
linkage_l = linkage(tsne_39D_l.drop(columns=['id']).values, method='ward')

num_clusters = 1000  # Adjust the number of clusters as needed
tsne_39D_s['Dendrogram_Cluster'] = fcluster(linkage_s, t=num_clusters, criterion='maxclust').astype(str)
tsne_39D_l['Dendrogram_Cluster'] = fcluster(linkage_l, t=num_clusters, criterion='maxclust').astype(str)

print("Generating 2D visualizations for linkage clusters...")
dendrogram_fig_s2STEP = px.scatter(
    tsne_2D_s, x='x', y='y', color=tsne_39D_s['Dendrogram_Cluster'],
    title="Linkage Clustering (Buffalo S - 39D t-SNE)"
)
dendrogram_fig_l2STEP = px.scatter(
    tsne_2D_l, x='x', y='y', color=tsne_39D_l['Dendrogram_Cluster'],
    title="Linkage Clustering (Buffalo L - 39D t-SNE)"
)
print("Linkage clustering complete.")


print("Beginning data preparation TSNE...")  
# Pre-compute t-SNE and PCA results
tsne_results_s3 = get_tsne_projection(buffalo_s_embed, perplexities=[3])[3]
tsne_results_l3 = get_tsne_projection(buffalo_l_embed, perplexities=[3])[3]
print("TSNE 3 complete.")
#tsne_results_s30 = get_tsne_projection(buffalo_s_embed, perplexities=[30])[30]
#tsne_results_l30 = get_tsne_projection(buffalo_l_embed, perplexities=[30])[30]
#print("TSNE 30 complete.")
tsne_results_s60 = get_tsne_projection(buffalo_s_embed, perplexities=[60])[60]
tsne_results_l60 = get_tsne_projection(buffalo_l_embed, perplexities=[60])[60]
print("TSNE 60 complete.")
tsne_results_s1000 = get_tsne_projection(buffalo_s_embed, perplexities=[1000])[1000]
tsne_results_l1000 = get_tsne_projection(buffalo_l_embed, perplexities=[1000])[1000]
print("TSNE 1000 complete.")
print("TSNE data preparation complete.")

    

#tsne_results_l_39D = get_tsne_projection(buffalo_l_embed, perplexities=[30], n_components=39)[30]
#tsne_results_s_39D = get_tsne_projection(buffalo_s_embed, perplexities=[30], n_components=39)[30]

#pca_results_s = get_pca_projection(buffalo_s_embed)
#pca_results_l = get_pca_projection(buffalo_l_embed)

# # Add IDs to dataframes
tsne_results_s3['id'] = buffalo_s_label['id']
tsne_results_l3['id'] = buffalo_l_label['id']
tsne_results_s30['id'] = buffalo_s_label['id']
tsne_results_l30['id'] = buffalo_l_label['id']
tsne_results_s60['id'] = buffalo_s_label['id']
tsne_results_l60['id'] = buffalo_l_label['id']
tsne_results_s1000['id'] = buffalo_s_label['id']
tsne_results_l1000['id'] = buffalo_l_label['id']

print("TSNE data preparation complete.")



#pca_results_s['id'] = buffalo_s_label['id']
#pca_results_l['id'] = buffalo_l_label['id']

print("Beginning clustering K-mean...")

# K-Means Clustering
kmeans_s = get_kmeans_clustering(tsne_results_s30[['x', 'y']].values, n_clusters=1000)
kmeans_l = get_kmeans_clustering(tsne_results_l30[['x', 'y']].values, n_clusters=1000)

# print(kmeans_s.shape)
# print(kmeans_l.shape)
# print(kmeans_s.head())
# print(kmeans_l.head())


kmeans_fig_s = px.scatter(kmeans_s, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_s)")
kmeans_fig_l = px.scatter(kmeans_l, x='x', y='y', color='Cluster', title="K-Means Clustering (buffalo_l)")
print("K-mean clustering complete.")

print("Beginning DBSCAN clustering...")
# # DBSCAN Clustering
dbscan_s = get_dbscan_clustering(tsne_results_s30[['x', 'y']].values, eps=3, min_samples=5)
dbscan_l = get_dbscan_clustering(tsne_results_l30[['x', 'y']].values, eps=3, min_samples=5)
dbscan_fig_s = px.scatter(dbscan_s, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_s)")
dbscan_fig_l = px.scatter(dbscan_l, x='x', y='y', color='Cluster', title="DBSCAN Clustering (buffalo_l)")
print("DBSCAN clustering complete.")


print("Beginning linkage generation...")
# Generate dendrogram images for t-SNE and PCA
dendrogram_image_tsne_s = create_dendrogram_plot(tsne_results_s30[['x', 'y']].values)
dendrogram_image_tsne_l = create_dendrogram_plot(tsne_results_l30[['x', 'y']].values)
#dendrogram_image_pca_s = create_dendrogram_plot(pca_results_s[['PCA1', 'PCA2']].values)
#dendrogram_image_pca_l = create_dendrogram_plot(pca_results_l[['PCA1', 'PCA2']].values)

linkage_s = linkage(tsne_results_s30.drop(columns=['id']).values, method='ward')  # Ward's method
linkage_l = linkage(tsne_results_l30.drop(columns=['id']).values, method='ward')

num_clusters = 1000  # Adjust the number of clusters as needed
tsne_results_s30['Dendrogram_Cluster'] = fcluster(linkage_s, t=num_clusters, criterion='maxclust').astype(str)
tsne_results_l30['Dendrogram_Cluster'] = fcluster(linkage_l, t=num_clusters, criterion='maxclust').astype(str)

print("Generating 2D visualizations for linkage clusters...")
dendrogram_fig_s2D = px.scatter(
    tsne_results_s30, x='x', y='y', color=tsne_results_s30['Dendrogram_Cluster'],
    title="Linkage Clustering (Buffalo S - 2D t-SNE)"
)
dendrogram_fig_l2D = px.scatter(
    tsne_results_l30, x='x', y='y', color=tsne_results_l30['Dendrogram_Cluster'],
    title="Linkage Clustering (Buffalo L - 2D t-SNE)"
)


print("Linkage generation complete.")
# Save everything to /datasets
data_to_save = {
    "buffalo_s": buffalo_s,
    "buffalo_l": buffalo_l,
    "tsne_results_s": tsne_results_s30,
    "tsne_results_l": tsne_results_l30,
    "tsne_results_s60": tsne_results_s60,
    "tsne_results_l60": tsne_results_l60,
    "tsne_results_s3": tsne_results_s3,
    "tsne_results_l3": tsne_results_l3,
    "tsne_results_s1000": tsne_results_s1000,
    "tsne_results_l1000": tsne_results_l1000,
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
    "tsne_39D_s": tsne_39D_s,
    "tsne_39D_l": tsne_39D_l,
    #"tsne_2D_s": tsne_2D_s,
    #"tsne_2D_l": tsne_2D_l,
    "kmeans_fig_s2STEP": kmeans_fig_s2STEP,
    "kmeans_fig_l2STEP": kmeans_fig_l2STEP,
    "dbscan_fig_s2STEP": dbscan_fig_s2STEP,
    "dbscan_fig_l2STEP": dbscan_fig_l2STEP,
    "dendrogram_fig_s2D": dendrogram_fig_s2D,
    "dendrogram_fig_l2D": dendrogram_fig_l2D,
    "dendrogram_fig_s2STEP": dendrogram_fig_s2STEP,
    "dendrogram_fig_l2STEP": dendrogram_fig_l2STEP
}

with open("./datasets/preprocessed_data.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print("Data preparation complete. All datasets saved to /datasets.")
