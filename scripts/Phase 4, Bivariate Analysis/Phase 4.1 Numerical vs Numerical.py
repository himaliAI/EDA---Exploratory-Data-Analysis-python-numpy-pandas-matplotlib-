# Pearson and Spearman correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Single pair analysis
corr_value = df[['mean radius', 'mean area']].corr(method='pearson')
print(f"Correlation value: {corr_value}")
sns.lmplot(x='mean radius', y='mean area', data=df, line_kws={'color': 'red'})
    # lmplot = linear model plot; combines scatterplot and regression line
    # line_kws lets you customize regression line's appearance: color, width, linestyle, transparency etc
plt.title("Mean Radius Vs Mean Area with Regression Line")
plt.show()

# Multiple pairs in one figure
pairs = ['mean radius', 'mean area', 'mean texture', 'mean perimeter', 'mean smoothness', 'mean compactness']
sns.pairplot(df[pairs], kind='reg', plot_kws={'line_kws': {'color': 'red'}}) # other kws eg scatter_kws
    # .pairplot, by default, shows scatterplots; we can change diagnonal plots with diag_kind
        # can add regression line wit kind='reg'
        # different colors for categorical variables with 'hue'
plt.suptitle("Scatterplots with Regression Lines for Selected Pairs", y=1.02)
    # .suptitle -> super title; y=1.02 shifts title upward to prevent overlap with plots
plt.show()

# compute correlation matrix
corr_matrix = df.corr(method='pearson')
print(corr_matrix.head())
# plot heatmap of correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title("Correlation Heatmap of Breast Cancer Features")
plt.show()

# filter strong correlation (>0.7)
strong_corr = corr_matrix[(corr_matrix > 0.7) & (corr_matrix < 1.0)]
print(f"Strong correlations:\n {strong_corr.dropna(how='all').dropna(axis=1, how='all')}")
    # first dropna(how='all') ~ dropna(axis=0, how='all') because axis=0 is the default; meaning -> drop row if all values are NaN
    # second dropna(axis=1, how="all") -> drop column if all values are NaN