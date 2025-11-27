# Skewness and Kurtosis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: "Malignant", 1: "Benign"})

# compute skewness and Kurtosis
skewness = df.drop(columns=['target', 'diagnosis']).skew()
kurtosis = df.drop(columns=['target', 'diagnosis']).kurtosis()

# sort by absolute skewness
skewness_sorted = skewness.abs().sort_values(ascending=False)
kurtosis_sorted = kurtosis.abs().sort_values(ascending=False)

# print top 10 skewed and kurtosis features
# print(f"Top 10 Skewness: {skewness_sorted.head(10)}")
# print(f"Top 10 Kurtosis: {kurtosis_sorted.head(10)}")

''' # one highly skewed/kurtosis distribution plot
# visualize 1 or 2 highly skewed/kurtosis features and plot their distribution
# eg area error for skewness and kurtosis
feature = 'area error'
plt.figure(figsize=(6,4))
sns.histplot(df[feature], kde=True, color='teal')
plt.title(f"Distribution of {feature} Skewness: {df[feature].skew():.2f}")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df[feature], kde=True, color='blue')
plt.title(f"Distribution of {feature} Kurtosis: {df[feature].kurtosis():.2f}")
plt.show()
'''

# plot histograms for top 10 skewed features
top_skewed_features = skewness_sorted.head(10).index.tolist()
    #select first 10 indexes (not values) and convert to python list
top_kurtosis_features = kurtosis_sorted.head(10).index.tolist()

plt.figure(figsize=(20,10))
for i, feature in enumerate(top_skewed_features, 1):
    plt.subplot(2, 5, i)
    sns.histplot(df[feature], kde=True, color='teal')
    plt.title(f"{feature} Skew: {df[feature].skew():.2f}")
plt.tight_layout
plt.show()

# similarly one can plot for top 10 kurtosis features