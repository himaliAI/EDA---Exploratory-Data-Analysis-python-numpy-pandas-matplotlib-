# Feature interaction
    # to see how two features together relate to the target (Interaction Visualization)
    # to create new features that capture combined effects
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: "Malignant", 1: "Benign"})

''' # create and interaction tern
df['radius_texture_interaction'] = df['mean radius'] * df['mean texture']
# compare distributions
sns.boxplot(x='diagnosis', y='radius_texture_interaction', data=df)
plt.title("Interaction term: Radius x Textue by Diagnosis")
plt.show()
    # if the distribution differ more clearly between groups, the interaction term adds value
'''

# Generalize interaction terms (multiple interaction terms)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df[['mean radius', 'mean texture', 'mean area']])
    # this generates all pairwise interactions automatically
# convert to DataFrame
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out())
print(interaction_df.head())

# interactions visualization
sns.scatterplot(x='mean radius', y='mean texture', hue='diagnosis', data=df)
plt.title("Interaction: Mean Radius Vs Mean Texture by diagnosis")
plt.show()
    # shows whether malignant vs benign tumors seperate more clearly when considering both features together
# Add size/style for a third feature
    # eg mean area as point size
sns.scatterplot(x='mean radius', y='mean texture', hue='diagnosis', size='mean area', sizes=(20, 200), data=df)
    # size -> each point's size is determined by size, 
    # sizes -> range of sizes for scaling, smallest size will be 20 and largest 200
plt.title("Radius vs Texture with Areas as Size")
plt.tight_layout()
plt.show()

# pairwise interaction grid
sns.pairplot(df[['mean radius', 'mean texture', 'mean area', 'diagnosis']], hue='diagnosis')
plt.show()
