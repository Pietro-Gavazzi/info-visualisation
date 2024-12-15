import pandas as pd
import numpy as np
import seaborn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels



buffalo_s = pd.read_csv("celeba_buffalo_s.csv")
buffalo_l = pd.read_csv("celeba_buffalo_l.csv")


embedding_names = []
for i in range(512):
    embedding_names.append("embedding_"+str(i))
    
buffalo_s_embed = buffalo_s[embedding_names]
buffalo_s_label = buffalo_s.drop(embedding_names, axis=1)

buffalo_l_embed = buffalo_l[embedding_names]
buffalo_l_label = buffalo_l.drop(embedding_names, axis=1)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(buffalo_l_embed)


pca = PCA(n_components=100)  # Example: Reduce to 2 components
principal_components = pca.fit_transform(scaled_data)
cumulative_variance = pca.explained_variance_ratio_.cumsum()

print(cumulative_variance)


df = buffalo_l_embed

# Compute the kernel matrix
kernel_matrix = pairwise_kernels(scaled_data, metric='sigmoid', gamma=0.1, coef0=1)

# Convert to DataFrame for better readability
kernel_df = pd.DataFrame(kernel_matrix, columns=[f"Point_{i+1}" for i in range(len(df))])
print(kernel_df)

# Example: Correlation between the first principal component and original features
pc1_correlations = pd.Series(
    [scaled_data[:, i].dot(principal_components[:, 0]) for i in range(scaled_data.shape[1])],
    index=df.columns
)
print(pc1_correlations)

from scipy.spatial.distance import pdist, squareform

# Pairwise Euclidean distances in the transformed space
transformed_distances = pdist(principal_components, metric='euclidean')

# Convert to a squareform matrix for better readability
distance_matrix = squareform(transformed_distances)
distance_df = pd.DataFrame(distance_matrix, columns=[f"Point_{i+1}" for i in range(len(df))])
print(distance_df)

import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot or scatterplot of the transformed space
sns.scatterplot(x=kpca_df['PC1'], y=kpca_df['PC2'])
plt.xlabel('Kernel Principal Component 1')
plt.ylabel('Kernel Principal Component 2')
plt.title('Transformed Space with Sigmoid Kernel')
plt.show()
