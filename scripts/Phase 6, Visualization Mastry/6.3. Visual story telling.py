# Narrative Dashboard Example
    # arrange three plots: scatterplot, Boxplot, Heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df =pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target # 0=malignant 1-benign
df['diagnosis'] = df['diagnosis'].map({0:"Malignant", 1:"Benign"})

# Figure with Gridspec layout
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# scatterplot at top-left
ax1 = fig.add_subplot(gs[0, 0])
sns.scatterplot(x="mean radius", y="mean texture", hue='diagnosis', data=df, ax=ax1, palette='Set1')
ax1.set_title("Scatterplot: Radius vs Texture")

# Boxplot at top right
ax2 = fig.add_subplot(gs[0, 1])
sns.boxplot(x="diagnosis", y="mean radius", data=df, hue='diagnosis', ax=ax2, palette='Set2')
ax2.set_title("Boxplot: Radius by Diagnosis")

# Heatmap at bottom both columns
ax3 = fig.add_subplot(gs[1, :])
corr = df[['mean radius', 'mean texture', 'mean area', 'mean perimeter']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()


