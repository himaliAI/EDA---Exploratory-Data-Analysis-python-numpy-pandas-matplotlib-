import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# heatmaps (show correlation or intensity values in grid)
corr = df.drop('diagnosis', axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# pairplot (Explore pairwise relationship across multiple variables)
sns.pairplot(df[['mean radius','mean texture','mean area','diagnosis']], hue='diagnosis')
plt.show()

# Catplots (compare categorical variables)
sns.catplot(x='diagnosis', y='mean radius', kind='box', data=df)
plt.show()

# Distribution plots (show feature distribuitons)
sns.histplot(df['mean radius'], kde=True, bins=30)
plt.title("Distribution of Mean Radius")
plt.show()

# Relational plots (show relationship between variables with extra encodings)
sns.relplot(x='mean radius', y="mean texture", hue='diagnosis', size='mean area', data=df)
plt.show()