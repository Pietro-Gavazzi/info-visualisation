import pandas as pd
import numpy as np
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

# Load data

def filterData():
    buffalo_s = pd.read_csv("celeba_buffalo_s.csv")
    buffalo_l = pd.read_csv("celeba_buffalo_l.csv")

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
    buffalo_s_filtered.to_csv("celeba_buffalo_s_reworked.csv", index=False)
    buffalo_l_filtered.to_csv("celeba_buffalo_l_reworked.csv", index=False)


    # Verify
    print("Number of common rows:", len(common_image_names))
    print ("Number of rows in buffalo_s_filtered:", len(buffalo_s_filtered))
    print ("Number of rows in buffalo_l_filtered:", len(buffalo_l_filtered))
filterData()