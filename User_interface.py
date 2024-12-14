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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

# Load data
buffalo_s = pd.read_csv("celeba_buffalo_s.csv")
buffalo_l = pd.read_csv("celeba_buffalo_l.csv")

# Extract embeddings and labels
embedding_names = [f"embedding_{i}" for i in range(512)]
buffalo_s_embed = buffalo_s[embedding_names]
buffalo_s_label = buffalo_s.drop(embedding_names, axis=1)

buffalo_l_embed = buffalo_l[embedding_names]
buffalo_l_label = buffalo_l.drop(embedding_names, axis=1)

np.sum(np.sum(buffalo_l_label.isna()))

buffalo_l_label

np.sum(np.sum(buffalo_l_embed.isna()))

buffalo_l_embed
import umap
n-blurry
indices1 = buffalo_l_label["Blurry"] == 1
indices0 = buffalo_l_label["Blurry"] == -1

v1 = np.sum(buffalo_l_embed[indices1])
v2 = np.sum(buffalo_l_embed[indices0])

V = np.matrix([v1, v2])
projection = np.linalg.inv(V @ V.T) @ V @ buffalo_l_embed.T
#print(projection[0][0])

# Prepare scatter plot data
scatter_data = []
scatter_data_blurry = []
scatter_data_non_blurry = []

# Loop through the indices where 'Blurry' is 1 or -1
for i in range(len(indices1)):
    if indices1[i]:  # Check if the index corresponds to 'Blurry'
        scatter_data.append((projection[i][0], projection[i][1], 'Blurry'))
        scatter_data_blurry.append((projection[i][0], projection[i][1], 'Blurry'))
for i in range(len(indices0)):
    if indices0[i]:
        scatter_data.append((projection[i][0], projection[i][1], 'Non-Blurry'))
        scatter_data_non_blurry.append((projection[i][0], projection[i][1], 'Non-Blurry'))

scatter_df = pd.DataFrame(scatter_data, columns=['x', 'y', 'Category'])
scatter_df_blurry = pd.DataFrame(scatter_data_blurry, columns=['x', 'y', 'Category'])
scatter_df_non_blurry = pd.DataFrame(scatter_data_non_blurry, columns=['x', 'y', 'Category'])


# Cosine similarity function
def cos_sim(v1, v2):
    return (v1.T @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Calculate cosine similarities for labels
vector_set = {}
for label in buffalo_l_label.columns.drop(["image_name", "id"]):
    indices_true = buffalo_l_label[label] == 1
    indices_false = buffalo_l_label[label] == -1

    vt = np.sum(buffalo_l_embed[indices_true], axis=0)
    vf = np.sum(buffalo_l_embed[indices_false], axis=0)

    vector_set[label] = [vt, vf]

similarities = []
label_names = buffalo_l_label.columns.drop(["image_name", "id"])
for i, label1 in enumerate(label_names):
    for label2 in label_names[i+1:]:
        sim = cos_sim(vector_set[label1][0], vector_set[label2][0])
        similarities.append((label1, label2, sim))

similarity_df = pd.DataFrame(similarities, columns=['Label1', 'Label2', 'Cosine Similarity'])
#print(similarity_df)

#selec part of scatter_df where Category is Blurry
Category = scatter_df['Category']
#print(Category)


embedding_values = buffalo_l_embed.values.flatten()


# t-SNE visualization

tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(buffalo_l_embed)
buffalo_l_embeded = pd.DataFrame({
    'tsne-2d-one': tsne_results[:, 0],
    'tsne-2d-two': tsne_results[:, 1],
    'Category': buffalo_l_label['Blurry'].replace({1: 'Blurry', -1: 'Non-Blurry'})
})

tsne_fig = px.scatter(
    buffalo_l_embeded, x='tsne-2d-one', y='tsne-2d-two', color='Category',
    title="t-SNE Projection of Embeddings",
    labels={'tsne-2d-one': 't-SNE X', 'tsne-2d-two': 't-SNE Y'}
)
tsne_fig.update_traces(marker=dict(size=6, opacity=0.6))

tsne_fig_list=[]
perplexity = [5,30,1000]
for i in range(len(perplexity)):
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity[i])
    tsne_results = tsne.fit_transform(buffalo_l_embed)
    buffalo_l_embeded = pd.DataFrame({
        'tsne-2d-one': tsne_results[:, 0],
        'tsne-2d-two': tsne_results[:, 1],
        'Category': buffalo_l_label['Blurry'].replace({1: 'Blurry', -1: 'Non-Blurry'})
    })

    tsne_fig_list.append(px.scatter(
    buffalo_l_embeded, x='tsne-2d-one', y='tsne-2d-two', color='Category',
    title="t-SNE Projection of Embeddings",
    labels={'tsne-2d-one': 't-SNE X', 'tsne-2d-two': 't-SNE Y'}
    ))
    tsne_fig_list[i].update_traces(marker=dict(size=6, opacity=0.6))
       
# Create bar plot of label occurrences
label_occurrences = buffalo_l_label.drop(columns=["image_name", "id"]).sum()#+len(buffalo_l_label)
label_occurrences_fig = px.bar(
    x=label_occurrences.index, y=label_occurrences.values,
    title="Occurrences of Each Label in Dataset",
    labels={'x': 'Label', 'y': 'Number of Occurrences'}
)
label_occurrences_fig.update_traces(marker_color='purple')



# Implement K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(buffalo_l_embed)

kmeans_fig = px.scatter(
    x=tsne_results[:, 0], y=tsne_results[:, 1], color=kmeans_labels.astype(str),
    title="K-Means Clustering Visualization",
    labels={'x': 't-SNE X', 'y': 't-SNE Y', 'color': 'Cluster'}
)
kmeans_fig.update_traces(marker=dict(size=6, opacity=0.6))

# Implement Hierarchical Clustering
linkage_matrix = linkage(buffalo_l_embed.sample(n=500, random_state=42), method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
hierarchical_fig = plt.gcf()
plt.close()

# Implement DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan_labels = dbscan.fit_predict(buffalo_l_embed)

dbscan_fig = px.scatter(
    x=tsne_results[:, 0], y=tsne_results[:, 1], color=dbscan_labels.astype(str),
    title="DBSCAN Clustering Visualization",
    labels={'x': 't-SNE X', 'y': 't-SNE Y', 'color': 'Cluster'}
)
dbscan_fig.update_traces(marker=dict(size=6, opacity=0.6))


# Linear DR: PCA
pca = PCA(n_components=2, random_state=0)
pca_results = pca.fit_transform(buffalo_l_embed)

pca_fig = px.scatter(
    x=pca_results[:, 0], y=pca_results[:, 1],
    color=buffalo_l_label['Blurry'].replace({1: 'Blurry', -1: 'Non-Blurry'}),
    title="PCA Projection of Embeddings",
    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
)
pca_fig.update_traces(marker=dict(size=6, opacity=0.6))

# Non-linear DR: UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_results = umap_reducer.fit_transform(buffalo_l_embed)

umap_fig = px.scatter(
    x=umap_results[:, 0], y=umap_results[:, 1],
    color=buffalo_l_label['Blurry'].replace({1: 'Blurry', -1: 'Non-Blurry'}),
    title="UMAP Projection of Embeddings",
    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
)
umap_fig.update_traces(marker=dict(size=6, opacity=0.6))









# Dash application
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
    html.H2("Proj blurry"),
    dcc.Graph(
        figure=px.scatter(
            scatter_df_blurry, x='x', y='y', color='Category',
            title="Blurry Projections",
            labels={'x': 'Projection X', 'y': 'Projection Y'}
        )
    ),
    html.H2("Proj non-blurry"),
    dcc.Graph(
        figure=px.scatter(
            scatter_df_non_blurry, x='x', y='y', color_discrete_sequence=['red'],
            title="Non-Blurry Projections",
            labels={'x': 'Projection X', 'y': 'Projection Y'}
        )
    ),
    html.H2("embeddings"),
    dcc.Graph(
        figure= px.histogram(
            embedding_values, nbins=50, 
            title="Distribution of Embedding Values",
            labels={'value': 'Embedding Value', 'count': 'Frequency'}
        )     
    ),
    html.H2("t-SNE Projection of Embeddings"),
    dcc.Graph(
        figure=tsne_fig
    ),
    
    html.H2("t-SNE Projection of Embeddings with different perplexity"),
    dcc.Graph(
        figure=tsne_fig_list[0]
    ),
    dcc.Graph(
        figure=tsne_fig_list[1]
    ),
    dcc.Graph(
        figure=tsne_fig_list[2]
    ),

    html.H2("Occurrences of Each Label in Dataset"),
    dcc.Graph(
        figure=label_occurrences_fig
    ),

    html.H2("K-Means Clustering"),
    dcc.Graph(
        figure=kmeans_fig
    ),

    html.H2("Hierarchical Clustering Dendrogram"),
    html.Div(
        children=dcc.Graph(figure=go.Figure(data=[])),  # Placeholder for dendrogram image
        style={"textAlign": "center"}
    ),

    html.H2("DBSCAN Clustering"),
    dcc.Graph(
        figure=dbscan_fig
    ),

    html.H2("PCA Projection of Embeddings"),
    dcc.Graph(
        figure=pca_fig
    ),

    html.H2("UMAP Projection of Embeddings"),
    dcc.Graph(
        figure=umap_fig
    )
   
    
    

    #,html.H2("Proj blurry"),
    #dcc.Graph(
    #    figure=px.imshow(
    #        similarity_df.pivot('Label1', 'Label2', 'Cosine Similarity').fillna(0),
    #        title="Cosine Similarity Matrix",
    #        labels={'color': 'Cosine Similarity'}
    #    )
    #)
])

if __name__ == '__main__':
    app.run_server(debug=True)