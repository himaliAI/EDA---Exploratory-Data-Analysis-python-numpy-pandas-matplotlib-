# Numerical Vs Categorical ananlysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

# group based summaries
group_summary = df.groupby('diagnosis')['mean radius'].describe()
    # group_summary -> count, mean, SD, min, 25%, 50%, 75%, max

sns.boxplot(x='diagnosis', y='mean radius', data=df)
plt.title("Mean Radius by Diagnosis")
plt.show()
sns.violinplot(x='diagnosis', y='mean radius', data=df)
plt.title("Mean Radius by Diagnosis (Violin plot)")
plt.show()

# ANOVA intuition
from scipy.stats import f_oneway
malignant_mean_radius = df[df['diagnosis'] == 'Malignant']['mean radius']
benign_mean_radius  = df[df['diagnosis'] == 'Benign']['mean radius']

f_stat, p_val = f_oneway(malignant_mean_radius, benign_mean_radius)
print(f"ANOVA f-statistics: {f_stat}")
print(f"p-value: {p_val}")

# t-test vs ANOVA
    # 1 sample t-test (ttest_1samp()) -> can use ANOVA -> No
    # independent t-test (ttest_rel()) -> which ANOVA -> one way ANOVA
    # paired t-test (ttest_ind()) -> which ANOVA -> repeated measured ANOVA