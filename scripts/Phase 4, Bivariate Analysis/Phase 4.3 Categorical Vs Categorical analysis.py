# create new categorical variable from numerical feature
# and build a categorical vs categorical analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from scipy.stats import chi2_contingency

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: "Malignant", 1: "Benign"})
df['radius_category'] = pd.qcut(df['mean radius'], q=3, labels=['small', 'medium', 'large'])
    # pd.qcut splits continuous data into q equal-sized groups (quantiles)(equal frequency)
    # vs pd.cut splits data into bins of equal range (intervals)

# contingency table
table = pd.crosstab(df['diagnosis'], df['radius_category'])
# chi-square test
chi2, p, dof, expected = chi2_contingency(table)

# visualization
sns.countplot(x="radius_category", hue="diagnosis", data=df)
plt.title("Diagnosis vs Radius Category")
plt.show()