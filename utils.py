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
from PIL import Image



back_to_menu_block = html.Div([
    html.A("Back to Menu", href="/", style={"margin-top": "20px", 'alignItems': 'center'})
], style={"justify-content": "center",'alignItems': 'center'})



       
# Function to convert image to base64 string
def encode_image(image_path):
    # Open image using Pillow
    img = Image.open(image_path)

    # Convert image to a byte stream
    buffered = BytesIO()
    img.save(buffered, format="JPEG")

    # Encode the byte stream to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{img_str}"




def get_tickals(percentages):
    # Define the range of y-values for the ticks
    min_y = min(percentages.values)
    max_y = max(percentages.values)

    # Define tick intervals (you can adjust this as needed)
    if max_y>50:
        tick_interval = 10  # Example: 10 percent intervals
    elif max_y>25:
        tick_interval = 5
    elif max_y>10:
        tick_interval = 2.5
    elif max_y>2.5:
        tick_interval = 1
    else:
        tick_interval=0.5

    # Generate a list of tick values from min_y to max_y with the defined interval
    return list(np.arange(11)*tick_interval)



def create_plotbar(dataset, xlabel="x", ylabel="y", title="My Plotbar"):
    # Create the scatter plot with lines
    fig = px.scatter(
        x=dataset.index,
        y=dataset.values,
        labels={"x": xlabel, "y": ylabel},
        title=title,
    )

    # Add points (dots) with custom styling
    fig.update_traces(
        mode="markers",  # Set mode to 'markers' for dots
        marker=dict(
            size=10,  # Size of the dots
            color="skyblue",  # Fill color of the dots
            line=dict(color="black", width=1)  # Black border around the dots
        )
    )

    # Add horizontal gray lines for each tick value
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=-0.5,
                x1=len(dataset) - 0.5,
                y0=z,
                y1=z,
                line=dict(color="skyblue", width=1)
            ) for z in get_tickals(dataset)
        ],
        xaxis=dict(title=xlabel, tickangle=45),
        yaxis=dict(title=ylabel, tickvals=get_tickals(dataset)),
        margin=dict(l=40, r=40, t=40, b=120),
        height=600,
        plot_bgcolor="white",
    )

    return fig





embedding_columns = ["embedding_"+str(i) for i in range(512)]
id_columns = ['id']
image_name_columns= ['image_name']
labels_columns = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
       'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
       'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
       'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
       'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
       'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
       'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
       'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
       'Wearing_Necktie', 'Young']


# Define functions to process and retrieve data
def load_data():
    df_s = pd.read_csv("datasets/celeba_buffalo_s_reworked.csv")
    df_l = pd.read_csv("datasets/celeba_buffalo_l_reworked.csv")
    return df_s, df_l

def preprocess_data(buffalo_s, buffalo_l):
    embedding_names = [f"embedding_{i}" for i in range(512)]
    buffalo_s_embed = buffalo_s[embedding_names]
    buffalo_s_label = buffalo_s.drop(embedding_names, axis=1)
    buffalo_l_embed = buffalo_l[embedding_names]
    buffalo_l_label = buffalo_l.drop(embedding_names, axis=1)
    return buffalo_s_embed, buffalo_s_label, buffalo_l_embed, buffalo_l_label

def get_projection(buffalo_l_embed, buffalo_l_label):
    indices1 = buffalo_l_label["Blurry"] == 1
    indices0 = buffalo_l_label["Blurry"] == -1

    v1 = np.sum(buffalo_l_embed[indices1].values, axis=0)
    v2 = np.sum(buffalo_l_embed[indices0].values, axis=0)

    V = np.vstack([v1, v2])
    projection = np.linalg.inv(V @ V.T) @ V @ buffalo_l_embed.values.T

    scatter_data = []
    for i in range(buffalo_l_embed.shape[0]):
        category = 'Blurry' if indices1.iloc[i] else 'Non-Blurry'
        scatter_data.append((projection[0, i], projection[1, i], category))

    return pd.DataFrame(scatter_data, columns=['x', 'y', 'Category'])

def get_cosine_similarity(buffalo_l_embed, buffalo_l_label):
    def cos_sim(v1, v2):
        return (v1.T @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    vector_set = {}
    label_names = buffalo_l_label.columns.drop(["image_name", "id"])

    for label in label_names:
        indices_true = buffalo_l_label[label] == 1
        indices_false = buffalo_l_label[label] == -1

        vt = np.sum(buffalo_l_embed[indices_true].values, axis=0)
        vf = np.sum(buffalo_l_embed[indices_false].values, axis=0)

        vector_set[label] = [vt, vf]

    similarities = []
    for i, label1 in enumerate(label_names):
        for label2 in label_names[i+1:]:
            sim = cos_sim(vector_set[label1][0], vector_set[label2][0])
            similarities.append((label1, label2, sim))

    return pd.DataFrame(similarities, columns=['Label1', 'Label2', 'Cosine Similarity'])

def get_tsne_projection(buffalo_l_embed, perplexities=[30],n_components=2,method='barnes_hut'):
    tsne_results = {}
    for perplexity in perplexities:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=0,method=method)
        tsne_proj = tsne.fit_transform(buffalo_l_embed)
        tsne_results[perplexity] = pd.DataFrame(tsne_proj, columns=['x', 'y'])
    return tsne_results

def get_label_occurrences(buffalo_l_label):
    label_occurrences = buffalo_l_label.drop(columns=["image_name", "id"]).sum()
    #get the % of each label
    #print(label_occurrences)
    label_occurrences = label_occurrences + len(buffalo_l_label)
    #print(label_occurrences)
    label_occurrences = label_occurrences / (0.02*len(buffalo_l_label))
    #print(label_occurrences)
    
    return label_occurrences


def get_pca_projection(buffalo_l_embed, n_components=2):
    pca = PCA(n_components, random_state=0)
    pca_results = pca.fit_transform(buffalo_l_embed)
    if n_components == 2:
        return pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    else:
        return pd.DataFrame(pca_results, columns=[f'PCA{i+1}' for i in range(n_components)])



def get_embed_projection(columns_label=labels_columns, column_not_label=labels_columns):
    dfl = pd.read_csv("datasets/l_embed.csv")
    dfl.drop("Unnamed: 0", axis=1, inplace=True)

    dfs = pd.read_csv("datasets/s_embed.csv")
    dfs.drop("Unnamed: 0", axis=1, inplace=True)

    columns_name = ["embed_"+i for i in columns_label]+["embed_not_"+i for i in column_not_label]
    return dfs[columns_name], dfl[columns_name]



def get_umap_projection(buffalo_l_embed):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(buffalo_l_embed)
    return pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])

def get_kmeans_clustering(tsne_results, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(tsne_results)
    #print(kmeans_labels)
    #print(kmeans_labels.shape)
    #handles K-mean in N dimensions and returns the cluster
    
    
    return pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'Cluster': kmeans_labels.astype(str)})

def get_hierarchical_clustering(buffalo_l_embed, sample_size=500):
    sampled_data = buffalo_l_embed.sample(n=sample_size, random_state=42)
    linkage_matrix = linkage(sampled_data, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    hierarchical_fig = plt.gcf()
    plt.close()
    return hierarchical_fig

def create_dendrogram_plot(buffalo_l_embed):
    buffalo_l_embed=pd.DataFrame(buffalo_l_embed)
    # Create a dendrogram plot as a static image
    linkage_matrix = linkage(buffalo_l_embed.sample(n=500, random_state=42), method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def get_dbscan_clustering(tsne_results, eps=3, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(tsne_results)
    return pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'Cluster': dbscan_labels.astype(str)})