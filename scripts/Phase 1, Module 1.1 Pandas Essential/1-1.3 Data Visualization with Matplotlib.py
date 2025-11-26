# Data visualization with Malplotlib
# Loading Breast cancer dataset in Pandas

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import seaborn as sns

# load dataset
cancer = load_breast_cancer() # data is a dictonary with .keys()
    # some imp keys are: data, target, target_names, feature_names, and DESCR

# convert to DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target # malignant = 0; benign = 1 (by convention)
df['target'] = 1 - df['target'] # now malignant = 1, benign = 0

# split data by diagnosis
malignant = df[df['target'] == 1]
benign = df[df['target'] == 0]

''' # Histograms
# plot histogram of mean radius grouped by target
plt.figure(figsize=(8,6))
malignant['mean radius'].hist(alpha=0.7, label='Malignant', bins=30, color='red')
benign['mean radius'].hist(alpha=0.7, label='Benign', bins=30, color='blue')
plt.xlabel("Mean Radius")
plt.ylabel('Frequency')
plt.title('Histogram of Tumor Mean Radius by Diagnosis')
plt.legend()
plt.show()

# plot histograms for other features eg 'mean texture', 'mean perimeter', 'mean area'
plt.figure(figsize=(8, 6))
malignant['mean texture'].hist(alpha=0.7, label="Malignant", bins=30, color='red')
benign['mean texture'].hist(alpha=0.6, label="Benign", bins=30, color='blue')
plt.xlabel("Mean texture")
plt.ylabel('Frequency')
plt.title("Histogram of Tumor Mean Textue by Diagnosis")
plt.legend()
plt.show()

# plot histogram for 'mean area'
plt.figure(figsize=(8, 6))
malignant['mean area'].hist(alpha=0.7, label="Malignant", bins=30, color='red')
benign['mean area'].hist(alpha=0.7, label="Benign", bins=30, color='blue')
plt.xlabel('Mean area')
plt.ylabel('Frequency')
plt.title("Histogram of Tumor Mean Area by Diagnosis")
plt.legend()
plt.show()
'''
''' # Scatter plots
# Scatter plots between Mean Radius and Mean Texture
plt.figure(figsize=(8, 6))
plt.scatter(malignant["mean radius"], malignant["mean texture"], color='red', alpha=0.6, label='Malignant')
plt.scatter(benign['mean radius'], benign['mean texture'], alpha=0.6, color='blue', label='Benign')
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.title("Scatter Plot: Mean Radius Vs Mean Texture")
plt.legend()
plt.show()

# Scatter plots between Mean Radius and Mean Area
plt.figure(figsize=(8, 6))
plt.scatter(malignant['mean radius'], malignant['mean area'], alpha=0.6, color='red', label="Malignant")
plt.scatter(benign['mean radius'], benign['mean area'], alpha=0.6, color='blue', label="Benign")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Area")
plt.title("Scatter Plot: Mean Radius Vs Mean Area")
plt.legend()
plt.show()
'''
''' # Seaborn's pairplot
# visualize multiple feature relationship with Seabon's pairplot
import seaborn as sns

# select a subset of features for clearity
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

# create a new dataframe with target included
df_subset = df[features + ['target']].copy()

# Map target to labels for readability
df_subset['diagnosis'] = df_subset['target'].map({1: 'Malignant', 0: "Benign"})

# pairplot
sns.pairplot(
    df_subset, 
    vars=features, 
    hue='diagnosis',                              # split by diagnosis
    palette={'Malignant':'red', 'Benign':'blue'}, # custom colors
    diag_kind='kde',                             # histograms on diagonal, option - kde/hist
    plot_kws={'alpha':0.6}                        # transparency for scatter points, other: s (marker size, 20), edgecolor, linewidth (thickness of marker edges)
)
plt.show()
'''
''' # pairplot for broader comparisions
# pairplot for broader comparision across dataset

# select first 10 features from the dataset
features = df.columns[:10].tolist()

# create a subset with target included
df_subset = df[features + ['target']].copy()

# map target to labels for readability
df_subset['diagnosis'] = df_subset['target'].map({0: 'Benign', 1: 'Malignant'})

# Pairplot
sns.pairplot(
    df_subset,
    vars=features,
    hue='diagnosis',
    palette={'Malignant': 'red', 'Benign': 'blue'},
    diag_kind='kde',
    plot_kws={'alpha': 0.6}
)
plt.show()
'''
''' # correlation matrix + heatmap
# correlation matrix + heatmap
corr = df.corr()

# heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr, 
    cmap='Spectral', # diverging (coolwarm, RdBu, Spectral, seismic); Sequential (Blues, Greens, Purples, Oranges etc)
    center=0,
    annot=False 
    #fmt=".2f" # format string to 2 decimal place    
)
plt.title("Correlation Heatmap fo Breast Cancer Features")
plt.show()
'''
''' # zooming in on correlations
# zooming in on correlations with the target column, and
    # rank features with the target column
corr_matrix = df.corr()

# extract correlations with target
target_corr = corr_matrix['target'].drop('target') # drop self-correlation
    # gives a Series of correlation values (-1 to 1) for each feature vs target

# rank features by absolute correlation strength
ranked_corr = target_corr.abs().sort_values(ascending=False)
print(ranked_corr)

# visualizing ranking with a bar plot
plt.figure(figsize=(10, 8))
ranked_corr.plot(kind='bar', color='teal')
plt.title("Feature Correlation Strength with Malignancy (Target)")
plt.ylabel("Absolute Correlation")
plt.show()
'''

# Boxplot (top predictors from correlation matrix:
    # 'mean radius', 'mean perimeter', 'mean area', 'mean concavity')
df['diagnosis'] = df['target'].map({0:'Benign', 1:'Malignant'})
top_features = ['mean radius', 'mean perimeter', 'mean area', 'mean concavity'] 
plt.figure(figsize=(12, 8))
for i, feature in enumerate(top_features, 1): # enumerate outputs index and values; 1 states start index from 1
    plt.subplot(2, 2, i)
    sns.boxplot(x='diagnosis', y=feature, data=df, hue='diagnosis',
                palette={'Malignant':'red', 'Benign':'blue'}, legend=False)
    plt.title(f"Boxplot of {feature} by Diagnosis")
plt.tight_layout() # avoid overlapping titles/labels
plt.show()

# Violin Plot
plt.figure(figsize=(12, 8))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i) # (nrow, ncol, which subplot to draw into counting left-to-right and top-to-bottom starting from 1)
    sns.violinplot(x='diagnosis', y=feature, data=df, hue='diagnosis', legend=False, 
                   palette={'Malignant':'red', 'Benign':'blue'}, split=True)
    plt.title(f"Violin Plot of {feature} by Diagnosis")
plt.tight_layout() # avoid overlapping titles/labels
plt.show()