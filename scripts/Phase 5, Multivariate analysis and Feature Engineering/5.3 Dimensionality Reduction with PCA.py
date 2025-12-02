import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Explained variance ratio
explained_var = pca.explained_variance_ratio_
    # In sklearn, attributes lerned during .fit() end with _
print(f"Explained variance ratio: {explained_var[:10]}") # first 10 components of 30

# Plot cumulative explained variance
plt.figure(figsize=(8,6))
plt.plot(range(1, len(explained_var)+1), explained_var.cumsum(), marker='o')
    # plot(1-30, cumsum of 30 vars)
plt.xlabel("Number of components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA explained variance")
plt.grid(True)
plt.show()

# visualize PCA components
# first two prinicpal components
df_pca = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
    # variances in X_pca is automatically arranged in decreasing order
df_pca['diagnosis'] = data.target
sns.scatterplot(x='PC1', y='PC2', hue='diagnosis', data=df_pca, palette='Set1')
plt.title("PCA Projection (PC1 vs PC2)")
plt.show()

# interprete PCA loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(len(pca.components_))], index=data.feature_names)
    # pca.components_ is a 2D array of principal axes of shape (principal components, original features)
    # Each row is one principal component 
    # Each column is weight of original feature in that component
    # .T transforms it to shape (original features, principal components)
    # set row index to original feature names
print(loadings.head())
