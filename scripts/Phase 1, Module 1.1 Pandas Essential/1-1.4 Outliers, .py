# Outlier Detection

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load breast cancer Wisconsin dataset from sklern
from sklearn.datasets import load_breast_cancer

# load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# add target column
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# outlier detection (IQR Method)
def detect_outliers_iqr(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1 
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers, lower_bound, upper_bound

''' # Outlier eg by IQR and boxplots
# Example: mean radius
outliers, lb, ub = detect_outliers_iqr(df, 'mean radius')
print(f"Outliers in mean radius: {len(outliers)}")
print(f"Lower bound: {lb}, Upper bound: {ub}")

# visualize outliers with boxplots
plt.figure(figsize=(6, 4))
sns.boxplot(x='diagnosis', y='mean radius', data=df, hue='diagnosis',
            palette={'Malignant': 'red', 'Benign': 'blue'})
plt.title("Boxplot of Mean Radius by Diagnosis")
plt.show() # outliers appears outside the whiskers
'''

# detect and summarize outliers across multiple top features
    # focusing on strongest predictors from correlation ranking
    # mean radius, mean perimeter, mean area, mean concavity
top_features = ['mean radius', 'mean perimeter', 'mean area', 'mean concavity']

# summary of outliers
outlier_summary = {}
for feature in top_features:
    count, lb, ub = detect_outliers_iqr(df, feature)
    outlier_summary[feature] = {
        'Outlier count': len(count),
        'Lower bound': lb,
        'Upper bound': ub
    }
print(outlier_summary)

# visualize outliers with boxplots
plt.figure(figsize=(12, 8))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='diagnosis', y=feature, data=df, hue='diagnosis',
                palette={'Malignant': 'red', 'Benign':'blue'}, legend=False)
    plt.title(f"Boxplot of {feature} by Diagnosis")
plt.tight_layout()
plt.show()