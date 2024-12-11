import pandas as pd
import numpy as np
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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


# Check for NaNs
#assert not buffalo_l_label.isna().any().any(), "NaN values found in labels"
#assert not buffalo_l_embed.isna().any().any(), "NaN values found in embeddings"

# Calculate projections for blurry and non-blurry
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
print(similarity_df)

#selec part of scatter_df where Category is Blurry
Category = scatter_df['Category']
#print(Category)


embedding_values = buffalo_l_embed.values.flatten()





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
            title="Blurry vs Non-Blurry Projections",
            labels={'x': 'Projection X', 'y': 'Projection Y'}
        )
    ),
    html.H2("Proj non-blurry"),
    dcc.Graph(
        figure=px.scatter(
            scatter_df_non_blurry, x='x', y='y', color_discrete_sequence=['red'],
            title="Blurry vs Non-Blurry Projections",
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